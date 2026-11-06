"""
Model registry: name → trainer class + the per-model default hyperparameters (taken from
each paper / official code, see the trainer docstrings and docs/BASELINES.md). Defaults
are applied to the shared argparse parser, so `--model X` alone reproduces X's recipe and
any flag can still be overridden on the command line.
"""
from .trainers.regression import RegressionTrainer

COMMON_DEFAULTS = dict(
    epochs=300, patience=40, batch_size=2, lr=1e-4, weight_decay=0.0, optimizer="adam",
    schedule="none", warmup_frac=0.1, amp=False, grad_clip=0.0, augment=False, flip_lr=True,
    intensity_jitter=True, num_workers=2, save_last=False,
    loss="masked_mse", init="default", rounds=10, round_epochs=15, round_lr_decay=0.5,
    round_schedule="",    # upstream_rounds refit: comma-separated epochs per round (no validation set)
    # regression / swin
    swin_feature_size=48, swin_drop_path=0.0, swin_checkpoint=False,
    # GAN family
    beta1=0.5, gan_mode="lsgan", gan_norm="batch", ngf=64, ndf=64, pool_size=0, n_const_epochs=100,
    label_smooth=False, use_dropout=True, num_downs=7, pad_shape="128,128,128",
    lambda_l1=100.0, lambda_sobel=100.0, rise_sobel=True, eagan_variant="g", pixel_loss_masked=False,
    # ct2pet
    n_in_slices=7, n_out_slices=3, pet_gamma=0.5, slab_stride=1, infer_batch=32,
    # jointsyn
    lr_g=2e-4, lr_d=2e-4, lr_cls=1e-4, lr_joint_g=1e-5, lr_joint_d=1e-6, lr_gamma_share=0.95,
    lr_gamma_cls=0.95, lr_gamma_joint=0.7, lambda_cycle=10.0, lambda_identity=0.3, lambda_cls=1.0,
    paired_l1=0.0, update_step=2, epochs_share=80, patience_share=20, epochs_cls=80, patience_cls=10,
    epochs_joint=15, cls_batch_size=2, joint_batch_size=1,
)

MODELS = {
    "denseunet": dict(
        trainer=RegressionTrainer,
        description="3D Dense-U-Net (Lee et al. 2024 port), Adam 1e-4, masked MSE",
        defaults=dict(epochs=300, patience=40, batch_size=2, lr=1e-4, weight_decay=0.0, optimizer="adam",
                      augment=False)),
    "denseunet_paper": dict(
        trainer=RegressionTrainer,
        description="3D Dense-U-Net exactly as released (Lee et al. 2024, tau_synthesis_train.py): Adam 1e-4, "
                    "batch 2, whole-volume MSE, glorot init, 10 rounds x <=15 epochs with best-weight reload "
                    "and lr x 0.5 between rounds, early stopping (patience 6) within each round",
        defaults=dict(epochs=150, patience=6, batch_size=2, lr=1e-4, weight_decay=0.0, optimizer="adam",
                      augment=False, schedule="upstream_rounds", loss="mse_full", init="glorot",
                      rounds=10, round_epochs=15, round_lr_decay=0.5)),
    "swinunetr": dict(
        trainer=RegressionTrainer,
        description="Swin UNETR (MONAI) regression head, AdamW 1e-4 wd 1e-5 warmup-cosine, AMP, masked MSE",
        defaults=dict(epochs=300, patience=40, batch_size=2, lr=1e-4, weight_decay=1e-5, optimizer="adamw",
                      schedule="warmup_cosine", warmup_frac=0.1, amp=True, augment=True)),
}


def model_names():
    return list(MODELS)


def defaults_for(model):
    d = dict(COMMON_DEFAULTS)
    d.update(MODELS[model]["defaults"])
    return d


def add_model_args(parser):
    """Register every hyperparameter with a None default; real defaults come from defaults_for()."""
    for k, v in COMMON_DEFAULTS.items():
        if isinstance(v, bool):
            parser.add_argument(f"--{k.replace('_', '-')}", dest=k, action="store_true", default=None)
            parser.add_argument(f"--no-{k.replace('_', '-')}", dest=k, action="store_false")
        else:
            parser.add_argument(f"--{k.replace('_', '-')}", dest=k, type=type(v), default=None)
    return parser


def resolve_args(args):
    """Fill unset (None) hyperparameters with the model's defaults."""
    for k, v in defaults_for(args.model).items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return args


def build_trainer(args, device, cls_map=None):
    return MODELS[args.model]["trainer"](args, device, cls_map=cls_map)
