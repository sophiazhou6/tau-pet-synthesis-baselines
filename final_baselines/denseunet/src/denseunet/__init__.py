"""
denseunet: 3D Dense-U-Net (Lee et al., Brain 2024), PyTorch port, and its training code.

  model.py         the network
  registry.py      model settings (`--model denseunet_paper` = the released recipe)
  trainers/        training loop: round schedule, early stopping, checkpoints
  cohort.py        cohort and CV folds (reuses TauGenNet's dataset class and split rule)
  config.py        cluster paths; puts TauGenNet on sys.path so its data loader and scorer are used
  predictions.py   packing of saved predictions
  gh_align.py      scoring helper, run under TauGenNet's code
  figs.py          figure helpers used by the scorer
"""
