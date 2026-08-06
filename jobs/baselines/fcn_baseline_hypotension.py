import os
import json
import yaml
import pandas as pd
from physiojepa.bedside import ForecastingDataset, CLIP_INTERPOLATE_RANGES
from pathlib import Path
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torch
from datetime import datetime
import numpy as np

from torch.utils.data import WeightedRandomSampler

from physiojepa.baselines import FCN, GeneralTimeSupervised, InceptionTime
from physiojepa.cross_channel_patchtst import SupervisedPatchTSTCrossChannel
from physiojepa.supervised_patchtst import SupervisedPatchTST
from sklearn.model_selection import StratifiedGroupKFold
from physiojepa.augmentations import MixupCallbackClassification, TransformsCallback, channel_masking, jitter_augmentation

from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.classification import AUROC, Accuracy, AveragePrecision
from functools import partial
from binary_metrics import bootstrap_binary_event_metrics, save_metrics_json
from resume_checkpoints import (
    build_resume_callbacks,
    find_resume_checkpoint,
    training_config_fingerprint,
)

torch.set_float32_matmul_precision('high')

config_path = os.environ.get('PHYSIOJEPA_CONFIG', 'fcn_baseline_hypotension.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

prepare_samples_only = os.environ.get('PHYSIOJEPA_PREPARE_SAMPLES_ONLY', '0') == '1'
random_state = config['run_config']['random_state']
pl.seed_everything(random_state, workers=True)
model_type = config['run_config'].get('model_type', 'fcn').lower()
if prepare_samples_only:
    model = None
elif model_type == 'fcn':
    model = FCN(**config['arch'])
elif model_type == 'inceptiontime':
    model = InceptionTime(**config['arch'])
elif model_type == 'supervised_patchtst':
    model = SupervisedPatchTST(**config['arch'])
elif model_type == 'supervised_patchtst_cross_channel':
    model = SupervisedPatchTSTCrossChannel(**config['arch'])
else:
    raise ValueError(
        f"Unsupported supervised model_type {model_type!r}; "
        "expected 'fcn', 'inceptiontime', 'supervised_patchtst', or "
        "'supervised_patchtst_cross_channel'"
    )


def save_sample_cache(sample_df, destination):
    """Atomically write a sample-index cache so interrupted jobs remain resumable."""
    destination = Path(destination)
    partial = destination.with_name(
        f".{destination.name}.{os.getpid()}.partial"
    )
    sample_df.to_csv(partial, compression='gzip', index=False)
    os.replace(partial, destination)


def summarize_cohort(datasets):
    """Summarize the exact post-waveform-filter cohort used by the model."""
    summary = {}
    label_column = f'outcome_val_{forecast_window_sec[0]}sec'
    for split, dataset in datasets.items():
        sample_df = dataset.sample_df
        counts = sample_df[label_column].value_counts().to_dict()
        summary[split] = {
            'samples': int(len(sample_df)),
            'positive_events': int(counts.get(1, 0)),
            'negative_events': int(counts.get(0, 0)),
            'icu_stays': int(sample_df['file_path'].nunique()),
            'patients': int(sample_df['subject_id'].nunique()),
        }
    return summary


def validate_cohort(summary, validation_config, label):
    """Compare an observed cohort with configured split-level expectations."""
    expected = validation_config.get('expected_cohort')
    if expected is None:
        return
    mismatches = []
    for split, expected_values in expected.items():
        observed_values = summary.get(split, {})
        for key, expected_value in expected_values.items():
            observed_value = observed_values.get(key)
            if observed_value != expected_value:
                mismatches.append(
                    f"{split}.{key}: observed {observed_value}, expected {expected_value}"
                )
    if mismatches:
        message = f"{label} cohort mismatch:\n  " + "\n  ".join(mismatches)
        if validation_config.get('strict_cohort_match', False):
            raise RuntimeError(message)
        print(f"WARNING: {message}")

forecast_window_sec = config['dataset']['forecast_window_sec']
if isinstance(forecast_window_sec, int):
    forecast_window_sec = [forecast_window_sec]

n_classes = config['arch'].get(
    'n_classes',
    config['arch'].get('num_classes', config['arch'].get('c_out')),
)
if n_classes is None:
    raise KeyError("arch must define 'n_classes', 'num_classes', or 'c_out'")
n_labels = len(forecast_window_sec)
if n_labels > 1:
    metrics = {"auroc":AUROC(task='multilabel', average='none', num_labels=n_labels),
            "auprc":AveragePrecision(task='multilabel', average='none', num_labels=n_labels),
            }
else:
    metrics = {"auroc":AUROC(task='binary'),
               "auprc":AveragePrecision(task='binary'),
            }

use_transforms = config['dataset']['use_transforms']
if use_transforms:
    transforms_callback = TransformsCallback(
        transforms=[
            partial(jitter_augmentation, mask_ratio=0.05, jitter_ratio=0.05, p=0.5), # jitter and 0 value timepoint masking
            partial(channel_masking, dim=1, p=0.1, specific_channels=None), # mask out channels with 0s
        ]
    )
else:
    transforms_callback = None

if config['training']['mixup']:
    mixup_callback = MixupCallbackClassification(num_classes=n_labels, mixup_alpha=config['training']['mixup_alpha'], ignore_index=2)
else:
    mixup_callback = None

BATCHSIZE = config['training']['batch_size']
num_workers = config['training']['num_workers']
data_dir = config['paths']['data_dir']
models_dir = os.environ.get(
    'PHYSIOJEPA_MODELS_DIR_OVERRIDE',
    config['paths']['models_dir'],
)
y_outcome = config['dataset']['y_outcome']

max_lr = config['scheduler']['max_lr']
div_factor = config['scheduler']['div_factor']
final_div_factor = config['scheduler']['final_div_factor']
pct_start=config['scheduler']['pct_start']
anneal_strategy=config['scheduler']['anneal_strategy']
scheduler_kwargs = dict(max_lr=max_lr, div_factor=div_factor, final_div_factor=final_div_factor, pct_start=pct_start, anneal_strategy=anneal_strategy)

perform_cv=config['training']['perform_cv']

outcome_df_path = config['paths']['outcome_df_path']
outcome_df = pd.read_csv(outcome_df_path)
outcome_df['Time Stamp (seconds)'] = outcome_df['Time Stamp (seconds)'].round()

smoke_config = config.get('smoke_test', {})
if smoke_config.get('enabled', False):
    subject_col = 'subject_id'
    if subject_col not in outcome_df:
        outcome_df[subject_col] = outcome_df['file_path'].apply(
            lambda path: Path(path).name.split('-', maxsplit=1)[0]
        )
    minimum_anchor_seconds = smoke_config.get(
        'minimum_anchor_seconds',
        config['dataset']['sample_seq_len_seconds'] + min(forecast_window_sec),
    )
    outcome_df = outcome_df.loc[
        outcome_df['Time Stamp (seconds)'] >= minimum_anchor_seconds
    ].copy()
    subject_labels = outcome_df.groupby(subject_col)[y_outcome].max()
    rng = np.random.default_rng(config['run_config']['random_state'])
    subjects_per_class = int(smoke_config['subjects_per_class'])
    positive_subjects = subject_labels.loc[subject_labels == 1].index.to_numpy()
    negative_subjects = subject_labels.loc[subject_labels == 0].index.to_numpy()
    if len(positive_subjects) < subjects_per_class or len(negative_subjects) < subjects_per_class:
        raise ValueError(
            f"Smoke subset needs {subjects_per_class} subjects per class, but found "
            f"{len(positive_subjects)} positive and {len(negative_subjects)} negative subjects"
        )
    positive_subjects = rng.choice(positive_subjects, subjects_per_class, replace=False)
    negative_subjects = rng.choice(negative_subjects, subjects_per_class, replace=False)
    positive_rows = outcome_df.loc[
        outcome_df[subject_col].isin(positive_subjects) & (outcome_df[y_outcome] == 1)
    ]
    negative_rows = outcome_df.loc[
        outcome_df[subject_col].isin(negative_subjects) & (outcome_df[y_outcome] == 0)
    ]
    max_rows = int(smoke_config['rows_per_subject'])
    sampled_groups = []
    for _, group in pd.concat([positive_rows, negative_rows]).groupby(subject_col, sort=True):
        sampled_groups.append(
            group.sample(
                n=min(max_rows, len(group)),
                random_state=config['run_config']['random_state'],
            )
        )
    outcome_df = pd.concat(sampled_groups, ignore_index=True)
    print(
        "Smoke subset:",
        len(outcome_df),
        "rows,",
        outcome_df[subject_col].nunique(),
        "subjects, labels",
        outcome_df[y_outcome].value_counts().sort_index().to_dict(),
    )

subject_split_path = config['paths'].get('subject_split_path')
if subject_split_path:
    subject_split = pd.read_csv(subject_split_path)
    required_split_columns = {'subject_id', 'split'}
    if not required_split_columns.issubset(subject_split.columns):
        raise ValueError(
            f"Subject split must contain {sorted(required_split_columns)}"
        )
    if subject_split['subject_id'].duplicated().any():
        raise ValueError("Subject split contains duplicate subject IDs")
    unexpected_splits = set(subject_split['split']) - {'train', 'val', 'test'}
    if unexpected_splits:
        raise ValueError(f"Unexpected split labels: {sorted(unexpected_splits)}")
    if 'subject_id' not in outcome_df:
        outcome_df['subject_id'] = outcome_df['file_path'].apply(
            lambda path: Path(path).name.split('-', maxsplit=1)[0]
        )
    split_subjects = {
        split: set(subject_split.loc[subject_split['split'] == split, 'subject_id'])
        for split in ('train', 'val', 'test')
    }
    if (
        split_subjects['train'] & split_subjects['val']
        or split_subjects['train'] & split_subjects['test']
        or split_subjects['val'] & split_subjects['test']
    ):
        raise ValueError("Subject leakage detected in fixed split manifest")
    outcome_df_train = outcome_df.loc[
        outcome_df['subject_id'].isin(split_subjects['train'])
    ].copy()
    outcome_df_val = outcome_df.loc[
        outcome_df['subject_id'].isin(split_subjects['val'])
    ].copy()
    outcome_df_test = outcome_df.loc[
        outcome_df['subject_id'].isin(split_subjects['test'])
    ].copy()
    print(
        "Loaded fixed subject split:",
        {split: len(subjects) for split, subjects in split_subjects.items()},
    )
else:
    zarr_files = outcome_df.file_path.tolist()
    groups = [Path(i).stem.split('-')[0] for i in zarr_files]
    labels = outcome_df[config['dataset']['y_outcome']].values.tolist()

    splitter = StratifiedGroupKFold(
        n_splits=10, shuffle=True, random_state=random_state
    )
    train_idxs, test_idxs = next(
        splitter.split(X=zarr_files, y=labels, groups=groups)
    )
    train_zarrs = [zarr_files[i] for i in train_idxs]
    test_zarrs = [zarr_files[i] for i in test_idxs]
    train_labels = [labels[i] for i in train_idxs]
    groups = [Path(i).stem.split('-')[0] for i in train_zarrs]
    splitter2 = StratifiedGroupKFold(
        n_splits=10, shuffle=True, random_state=random_state
    )
    train_idxs, val_idxs = next(
        splitter2.split(X=train_zarrs, y=train_labels, groups=groups)
    )
    val_zarrs = [train_zarrs[i] for i in val_idxs]
    train_zarrs = [train_zarrs[i] for i in train_idxs]
    outcome_df_train = outcome_df.loc[
        outcome_df.file_path.isin(train_zarrs)
    ].copy()
    outcome_df_val = outcome_df.loc[
        outcome_df.file_path.isin(val_zarrs)
    ].copy()
    outcome_df_test = outcome_df.loc[
        outcome_df.file_path.isin(test_zarrs)
    ].copy()


channels = config['dataset']['channels']
c_in = len(channels)

frequency = config['dataset']['frequency']

sample_seq_len_seconds = config['dataset']['sample_seq_len_seconds']
win_length = config['dataset']['win_length'] # every ten seconds for patches
overlap = config['dataset']['overlap']
hop_length=win_length - int(overlap*win_length)
max_seq_len = sample_seq_len_seconds*frequency


dataset_filename = config['paths']['dataset_filename']
os.makedirs(models_dir, exist_ok=True)

n_patches = (max(max_seq_len, win_length)-win_length) // hop_length + 1
if ((max_seq_len-win_length) % hop_length != 0):
    n_patches += 1
n_patches = int(n_patches)

sample_cache_dir = Path(config['paths'].get('sample_cache_dir', models_dir))
train_cache_path = sample_cache_dir / f'{dataset_filename}-train_samples.csv.gz'
val_cache_path = sample_cache_dir / f'{dataset_filename}-val_samples.csv.gz'
test_cache_path = sample_cache_dir / f'{dataset_filename}-test_samples.csv.gz'

if train_cache_path.exists():
    sample_df_train = pd.read_csv(train_cache_path)
else:
    sample_df_train = None
if val_cache_path.exists():
    sample_df_val = pd.read_csv(val_cache_path)
else:
    sample_df_val = None
if test_cache_path.exists():
    sample_df_test = pd.read_csv(test_cache_path)
else:
    sample_df_test = None

if config['training'].get('require_precomputed_samples', False) and not prepare_samples_only:
    missing_caches = [
        str(path)
        for path in (train_cache_path, val_cache_path, test_cache_path)
        if not path.exists()
    ]
    if missing_caches:
        raise FileNotFoundError(
            "Run the CPU sample-index sbatch stage first. Missing caches:\n  "
            + "\n  ".join(missing_caches)
        )

y_padding_mask = config['dataset']['y_padding_mask']
constant_nan_tolerance = config['dataset']['constant_nan_tolerance']
require_all_channels = config['dataset']['require_all_channels']
infer_forecast_windows = config['dataset']['infer_forecast_windows']
 

name = config['run_config']['name']
filename = f"{name}" + "{epoch:02d}-Focal:{val_loss:.5f}-CE:{val_ce_loss:.5f}"
date_str = datetime.now().strftime("%Y-%m-%d")
resume_config = config.get('resume', {})
sub_dir = resume_config.get(
    'run_subdir',
    f"{date_str}-{config['run_config']['model_type']}-{config['run_config']['model_run']}",
)
checkpoint_dir = Path(models_dir) / sub_dir
checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=1, monitor="val_loss", mode='min', filename=filename)
checkpoint_callback2 = ModelCheckpoint(
    dirpath=checkpoint_dir,
    save_top_k=1,
    monitor="train_loss",
    mode='min',
    filename=filename,
    save_last=True,
    enable_version_counter=False,
)
checkpoint_callback2.CHECKPOINT_NAME_LAST = "epoch-last"
checkpoint_callback3 = ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=1, monitor="val_auprc_0" if n_labels > 1 else "val_auprc", mode='max', filename=filename)
checkpoints = [checkpoint_callback, checkpoint_callback2, checkpoint_callback3]
config_fingerprint = training_config_fingerprint(config)
resume_checkpoint_path = None
if resume_config.get('enabled', False):
    metadata_callback, rolling_checkpoint_callback = build_resume_callbacks(
        checkpoint_dir=checkpoint_dir,
        config_fingerprint=config_fingerprint,
        run_subdir=sub_dir,
        checkpoint_interval_minutes=float(
            resume_config.get('checkpoint_interval_minutes', 30)
        ),
    )
    checkpoints.extend([metadata_callback, rolling_checkpoint_callback])
    configured_initial_checkpoint = resume_config.get('initial_checkpoint_path')
    resume_checkpoint_path = find_resume_checkpoint(
        checkpoint_dir=checkpoint_dir,
        expected_config_fingerprint=config_fingerprint,
        initial_checkpoint_path=(
            Path(configured_initial_checkpoint)
            if configured_initial_checkpoint
            else None
        ),
        allow_unverified_initial_checkpoint=resume_config.get(
            'allow_unverified_initial_checkpoint',
            False,
        ),
    )
    if resume_checkpoint_path is None:
        print(f"No compatible resume checkpoint found in {checkpoint_dir}; starting from epoch 0")
    else:
        print(f"Resuming complete training state from {resume_checkpoint_path}")
wandb_project = config['run_config']['wandb_project']
wandb_name = f"{sub_dir}-{config['run_config']['name']}"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

if prepare_samples_only:
    wandb_logger = None
else:
    wandb_logger = WandbLogger(
        project=f"{wandb_project}",
        offline=config['run_config'].get('wandb_offline', False),
        name=wandb_name,
        save_dir=str(checkpoint_dir),
    )
    wandb_logger.log_hyperparams(config)

if __name__ == "__main__":
    train_ds = ForecastingDataset(
                 channels=channels, 
                 forecast_window_sec = forecast_window_sec,
                 outcome_df = outcome_df_train,
                 outcome_df_outcome_col=y_outcome,
                 file_col='file_path',
                 y_date_column='date', # column indicating date of sample collection
                 outcome_df_seconds_since_column='Time Stamp (seconds)', # column indicating how many seconds since beginning of waveform
                 outcome_df_duration_column='event_length',
                 sample_df = sample_df_train,
                 sample_seq_len_sec=sample_seq_len_seconds, 
                 frequency=frequency, 
                 butterworth_filters=None,
                 median_filter_kernel_size=None,
                 clip_interpolations=CLIP_INTERPOLATE_RANGES,
                 constant_nan_tolerance=constant_nan_tolerance,
                 require_all_channels=True,
                 infer_forecast_windows=infer_forecast_windows,
                 normalize_signals=config['dataset']['normalize_signals'],
                 sample_generation_workers=config['training'].get('sample_generation_workers')
    )
    val_ds = ForecastingDataset(
                 channels=channels, 
                 forecast_window_sec = forecast_window_sec,
                 outcome_df = outcome_df_val,
                 outcome_df_outcome_col=y_outcome,
                 file_col='file_path',
                 y_date_column='date', # column indicating date of sample collection
                 outcome_df_seconds_since_column='Time Stamp (seconds)', # column indicating how many seconds since beginning of waveform
                 outcome_df_duration_column='event_length',
                 sample_df = sample_df_val,
                 sample_seq_len_sec=sample_seq_len_seconds, 
                 frequency=frequency, 
                 butterworth_filters=None,
                 median_filter_kernel_size=None,
                 clip_interpolations=CLIP_INTERPOLATE_RANGES,
                 constant_nan_tolerance=constant_nan_tolerance,
                 require_all_channels=True,
                 infer_forecast_windows=infer_forecast_windows,
                 normalize_signals=config['dataset']['normalize_signals'],
                 sample_generation_workers=config['training'].get('sample_generation_workers')
    )

    test_ds = ForecastingDataset(
                    channels=channels, 
                    forecast_window_sec = forecast_window_sec,
                    outcome_df = outcome_df_test,
                    outcome_df_outcome_col=y_outcome,
                    file_col='file_path',
                    y_date_column='date', # column indicating date of sample collection
                    outcome_df_seconds_since_column='Time Stamp (seconds)', # column indicating how many seconds since beginning of waveform
                    outcome_df_duration_column='event_length',
                    sample_df=sample_df_test,
                    sample_seq_len_sec=sample_seq_len_seconds, 
                    frequency=frequency, 
                    butterworth_filters=None,
                    median_filter_kernel_size=None,
                    clip_interpolations=CLIP_INTERPOLATE_RANGES,
                    constant_nan_tolerance=constant_nan_tolerance,
                    require_all_channels=True,
                    infer_forecast_windows=infer_forecast_windows,
                    normalize_signals=config['dataset']['normalize_signals'],
                    sample_generation_workers=config['training'].get('sample_generation_workers')
    )
    
    if not train_cache_path.exists():
        save_sample_cache(train_ds.sample_df, train_cache_path)
    if not val_cache_path.exists():
        save_sample_cache(val_ds.sample_df, val_cache_path)
    if not test_cache_path.exists():
        save_sample_cache(test_ds.sample_df, test_cache_path)

    datasets = {'train': train_ds, 'val': val_ds, 'test': test_ds}
    if subject_split_path:
        for split, dataset in datasets.items():
            observed_subjects = set(dataset.sample_df['subject_id'].unique())
            if observed_subjects != split_subjects[split]:
                missing = sorted(split_subjects[split] - observed_subjects)
                unexpected = sorted(observed_subjects - split_subjects[split])
                raise RuntimeError(
                    f"{split} cache does not match fixed subject manifest: "
                    f"{len(missing)} missing, {len(unexpected)} unexpected"
                )
    cohort_summary = summarize_cohort(datasets)
    cohort_summary_path = Path(models_dir) / f'{dataset_filename}-cohort_summary.json'
    save_metrics_json(cohort_summary, cohort_summary_path)
    print("Post-filter cohort:", json.dumps(cohort_summary, sort_keys=True))
    validate_cohort(
        cohort_summary,
        config.get('cohort_validation', {}),
        "Fixed split",
    )
    validate_cohort(
        cohort_summary,
        config.get('paper_replication', {}),
        "Published Table 1",
    )

    if prepare_samples_only:
        print(f"Sample preparation complete: {cohort_summary_path}")
        raise SystemExit(0)

    label_weights = 1 / train_ds.sample_df[f'outcome_val_{forecast_window_sec[0]}sec'].value_counts(dropna=False, normalize=False)
    sample_weights = [label_weights[i] for i in train_ds.sample_df[f'outcome_val_{forecast_window_sec[0]}sec'].values.tolist()]
    weighted_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds), replacement=True)

    persistent_workers = num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=BATCHSIZE, sampler=weighted_sampler, drop_last=True, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCHSIZE, shuffle=False, drop_last=False, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=BATCHSIZE, shuffle=False, drop_last=False, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=False)
    
    if config['training']['use_class_weights']:
        class_weights = []
        for forecast_sec in forecast_window_sec:
            class0 = train_ds.sample_df[f'outcome_val_{forecast_sec}sec'].value_counts(dropna=False, normalize=False).sort_index().values[0]
            class1 = train_ds.sample_df[f'outcome_val_{forecast_sec}sec'].value_counts(dropna=False, normalize=False).sort_index().values[1]
            pos_weight = class0 / class1
            class_weights.append(pos_weight)
        class_weights = torch.tensor(class_weights)
    else:
        class_weights = None
   
    patchmeupe2e_model = GeneralTimeSupervised(supervised_model=model,
                 learning_rate=config['optimizer']['learning_rate'],
                 train_size=len(train_ds),
                 batch_size=BATCHSIZE,
                 n_gpus=config['training']['n_gpus'],
                 n_classes=1, # this should be 1 for binary classification / multilabel classification
                 n_labels=1 if n_classes != 1 else n_labels, # n labels is the number of forecast windows, which corresponds to levels in the hierarchical inception time architecture
                 metrics=metrics, # name:function for metrics to log
                 loss_fxn=config['training']['loss_fxn'], # loss function to use, can be CrossEntropy or FocalLoss
                 gamma=config['training']['gamma'], # for focal loss
                 class_weights=class_weights, # weights of classes to use in CE loss fxn
                 label_smoothing=config['training']['label_smoothing'], # label smoothing for cross entropy loss
                 y_padding_mask=y_padding_mask, # padded value that was added to target and indice to ignore when computing loss
                 epochs=config['training']['epochs'], # number of epochs for one_cycle_scheduler
                 optimizer_type=config['optimizer']['optimizer_type'],
                 scheduler_type=config['scheduler']['scheduler_type'],
                 weight_decay=config['optimizer']['weight_decay'],
                 use_weight_decay_scheduler=config['optimizer']['use_weight_decay_scheduler'],
                 final_weight_decay=config['optimizer']['final_weight_decay'],
                 transforms=transforms_callback,
                 mixup_callback=mixup_callback,
                 scheduler_kwargs=scheduler_kwargs,
                 )
    
    training_strategy = "ddp" if config['training']['n_gpus'] > 1 else "auto"
    use_sync_batchnorm = config['training']['n_gpus'] > 1
    trainer = pl.Trainer(precision=config['run_config']['precision'],
                        enable_checkpointing=True, 
                        enable_progress_bar=True, 
                        enable_model_summary=True, 
                        logger=wandb_logger,
                        val_check_interval=config['training']['val_check_interval'],
                        log_every_n_steps=50,
                        num_sanity_val_steps=2, # speed up
                        detect_anomaly=False, # speed up, though defualt
                        deterministic=config['run_config'].get('deterministic', False),
                        profiler=None, # this is the default, not me
                        strategy=training_strategy,
                        gradient_clip_val=config['training']['gradient_clip_val'],
                        gradient_clip_algorithm='norm' if config['training']['use_gradient_clipping'] else None,
                        accelerator="gpu", 
                        devices=config['training']['n_gpus'], 
                        default_root_dir=str(checkpoint_dir),
                        max_epochs=config['training']['epochs'], 
                        fast_dev_run=False,
                        limit_train_batches=config['training'].get('limit_train_batches', 1.0),
                        limit_val_batches=config['training'].get('limit_val_batches', 1.0),
                        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
                        sync_batchnorm=use_sync_batchnorm,
                        callbacks=checkpoints
                        )


    trainer.fit(
        model=patchmeupe2e_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=str(resume_checkpoint_path) if resume_checkpoint_path else None,
    )
    del trainer
    #Predictions and targets
    trainer2 = pl.Trainer(precision=config['run_config']['precision'],
                        enable_checkpointing=False, 
                        enable_progress_bar=True, 
                        enable_model_summary=False, 
                        logger=None,
                        val_check_interval=config['training']['val_check_interval'],
                        log_every_n_steps=None,
                        num_sanity_val_steps=0, # speed up
                        detect_anomaly=False, # speed up, though defualt
                        deterministic=config['run_config'].get('deterministic', False),
                        profiler=None, # this is the default, not me
                        strategy="auto",
                        gradient_clip_val=config['training']['gradient_clip_val'],
                        gradient_clip_algorithm='norm' if config['training']['use_gradient_clipping'] else None,
                        accelerator="gpu", 
                        devices=1, # only 1 device for inference
                        default_root_dir=str(checkpoint_dir),
                        max_epochs=config['training']['epochs'], 
                        fast_dev_run=False,
                        limit_predict_batches=config['training'].get('limit_predict_batches', 1.0),
                        sync_batchnorm=False,
                        )
    
    best_model_path = checkpoint_callback3.best_model_path
    val_preds, val_targets = list(zip(*trainer2.predict(model=patchmeupe2e_model, dataloaders=val_loader, return_predictions=True, ckpt_path=best_model_path)))
    test_preds, test_targets = list(zip(*trainer2.predict(model=patchmeupe2e_model, dataloaders=test_loader, return_predictions=True, ckpt_path=best_model_path)))
    
    test_targets_cat = torch.cat(test_targets).cpu()
    test_preds_cat = torch.cat(test_preds).cpu()

    val_preds_cat = torch.cat(val_preds).cpu()
    val_targets_cat = torch.cat(val_targets).cpu()

    
    tensor_dict = {'val_targets': val_targets_cat, 'val_preds': val_preds_cat,
                   'test_targets': test_targets_cat, 'test_preds': test_preds_cat,
                   }
    prediction_path = os.path.join(models_dir, sub_dir, f'{config["dataset"]["y_outcome"]}-{config["run_config"]["name"]}-predictions.pt')
    torch.save(tensor_dict, prediction_path)

    evaluation_config = config.get('evaluation', {})
    if evaluation_config.get('paper_metrics', True):
        bootstrap_resamples = int(evaluation_config.get('bootstrap_resamples', 1000))
        bootstrap_seed = int(evaluation_config.get('bootstrap_seed', random_state))
        paper_metrics = {
            'prediction_path': prediction_path,
            'checkpoint_path': best_model_path,
            'validation': bootstrap_binary_event_metrics(
                val_targets_cat.numpy(),
                val_preds_cat.numpy(),
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            ),
            'test': bootstrap_binary_event_metrics(
                test_targets_cat.numpy(),
                test_preds_cat.numpy(),
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            ),
        }
        metrics_path = os.path.join(
            models_dir,
            sub_dir,
            f'{config["dataset"]["y_outcome"]}-{config["run_config"]["name"]}-paper_metrics.json',
        )
        save_metrics_json(paper_metrics, metrics_path)
        print(f"Paper-style metrics saved to {metrics_path}")
    
    auroc_metric = AUROC(task='binary')
    avg_prec_metric = AveragePrecision(task='binary')
    accuracy_metric = Accuracy(task='binary')

    # print(auroc_scores)
    print("Val AUC", auroc_metric(val_preds_cat, val_targets_cat))
    print("Val AP", avg_prec_metric(val_preds_cat, val_targets_cat))
    print("Val Accuracy", accuracy_metric(val_preds_cat, val_targets_cat))

    print('Test AUC', auroc_metric(test_preds_cat, test_targets_cat))
    print('Test AP', avg_prec_metric(test_preds_cat, test_targets_cat))
    print('Test Accuracy', accuracy_metric(test_preds_cat, test_targets_cat))
