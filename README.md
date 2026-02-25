# hgm_clean (baseline clean)

This folder is a *minimal cleaned* snapshot of your current codebase:
- same core logic as before (teacher/estimators/measures),
- fewer files,
- duplicate function definitions removed,
- canonical runner names.

## Files
Core modules (importable):
- teacher.py
- estimators.py
- measures.py

Canonical runners:
- run_2layers_nmse.py
- run_3layers_nmse.py
- run_2layers_spectrum.py
- run_spectrum_slq.py
- run_h3_nmse.py
- run_h3_spectrum.py

## Quick smoke test
```python
import torch
import teacher, estimators, measures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean_y, std_y = teacher.compute_mean_std_y_stream(d=50, p=10, n=2000, batch_size=512, seed=0, device=device)
print(mean_y, std_y)
```