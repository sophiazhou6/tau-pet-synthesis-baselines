# tau-pet-synthesis-baselines

Two published MRI→tau-PET synthesis models, run as baselines on an expanded ADNI cohort:

- **3D Dense-U-Net** (MRI only). A PyTorch port of [Neurology-AI-Program/AI_imputed_tau_PET](https://github.com/Neurology-AI-Program/AI_imputed_tau_PET), the code accompanying Lee et al., ["Synthesizing images of tau pathology from cross-modal neuroimaging using deep learning"](https://pubmed.ncbi.nlm.nih.gov/37804318/), *Brain* 147(3):980–995, 2024.
- **TauGenNet** (MRI + plasma p-tau217). A 3D latent diffusion model conditioned through a frozen CLIP text encoder. After Gong et al., ["TauGenNet: Plasma-Driven Tau PET Image Synthesis via Text-Guided 3D Diffusion Models"](https://arxiv.org/abs/2509.04269), IEEE TRPMS.

**The code for the current runs is in [`final_baselines/`](final_baselines/).** Its README lists where each model,
training script and job script is.

[`archive/`](archive/) holds earlier code that the current runs do not use: the summer 2026 code and results, and
the first upload of the November runs.

No ADNI data, subject IDs, checkpoints or predictions are included.
