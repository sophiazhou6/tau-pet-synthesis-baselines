"""
training — model-agnostic MRI → tau-PET synthesis baselines for the TauGenNet manuscript.

Every model here is trained, generated and scored through the same three entry points
(scripts/train_baseline.py, scripts/generate_baseline.py, scripts/score_baseline.py) on a
manifest-driven cohort (training/cohort.py) that reuses TauGenNet's dataset class and
split policy, and scored with TauGenNet's evaluate_final.py metric functions — so every
number is directly comparable to the TauGenNet / Dense-U-Net results already in the paper.

Model families (see docs/BASELINES.md for the per-model cards):
  denseunet   3D Dense-U-Net (Lee et al., Brain 2024) — the existing deterministic baseline
  swinunetr   Swin UNETR (Hatamizadeh et al., 2022) with a linear regression head
  eagan       Ea-GAN — gEa-GAN / dEa-GAN (Yu et al., IEEE TMI 2019), 3D pix2pix + Sobel edge loss
  ct2pet      2.5D conditional GAN with ResUNet++ generator (Salehjahromi et al., Cell Rep Med 2024)
  jointsyn    Joint synthesis + diagnosis framework, ShareGAN + AwareNet (Wang et al., MedIA 2024)
"""
