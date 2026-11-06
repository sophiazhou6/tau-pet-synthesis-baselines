# Archive

Earlier code, kept for reference. The current runs use [`../final_baselines/`](../final_baselines/).

| Folder | What it is |
|---|---|
| [`summer-2026/`](summer-2026/) | The summer 2026 code and its results (`RESULTS.md`). Those results come from the earlier cohort and used a left-right mirrored atlas, so they are not comparable with the current runs. The git tag `summer-2026` has this code at its original paths. |
| [`updated_runs/`](updated_runs/) | The first upload of the November 2026 runs (commit 7be8555). It has the Dense-U-Net and pipeline code, and the TauGenNet changes as a patch against `summer-2026/taugennet/`. `final_baselines/` holds the parts of it the runs used, with the patch applied. The figure scripts (`scripts/make_composite_figure.py`, `scripts/render_glass_all_subjects.py`) and the checks behind the results (`audit/`) are only here. Paths in its README refer to the layout before it was moved. |
