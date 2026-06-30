# U-DICNet

Unsupervised DIC displacement measurement from a single speckle image pair.

Based on: *"Unsupervised Deep Learning DIC with only Single-pair image for high order displacement measurement"*

## Requirements

```bash
pip install -r requirements.txt
```

Tested on Python ≥ 3.7, PyTorch ≥ 1.7.

## Quick start

```bash
# default: U_DICNet (2-ch output) + patch_grad loss
python main.py --data-dir ./gauss_displacement
```

## Arguments

| argument | default | description |
|---|---|---|
| `--data-dir` | `./gauss_displacement` | directory with `re*.bmp` / `tar*.bmp` |
| `--arch` | `U_DICNet` | network: `U_DICNet` (2-ch) or `U_DICNet_shape2` (12-ch) |
| `--loss` | `patch_grad` | `patch_grad` = 2-ch + numerical gradients; `patch12` = 12-ch Taylor |
| `--pretrained` | `None` | path to checkpoint `.pth.tar` |
| `--solver` | `adam` | `adam` or `sgd` |
| `--lr` | `0.001` | initial learning rate |
| `--epochs` | `2500` | total epochs |
| `--radius` | `2` | subset radius (pixels) |
| `--order` | `2` | Taylor expansion order |
| `--norm-factor` | `10.0` | image normalisation = pixel / 255 × factor |
| `--save-dir` | same as `--data-dir` | output directory |
| `--seed` | `42` | random seed for reproducibility |
| `--auto-retry` | `False` | enable adaptive lr retry on stalled convergence |
## Convergence strategies
| argument | default | description |
| --- | --- | --- |
| `--warmup N` | `0` (off) | linear lr warmup: ramp from 1e-7 to `--lr` over N epochs |
| `--reverse-lr` | `False` | adaptive lr: raise (×2) when loss drops >10%, lower (÷2) when <0.5% per 50 epochs. Starts at 1e-6, ignores ReduceLROnPlateau |
| `--auto-retry` | `False` | auto-detect stalled/diverging runs: re-initialises model, lower lr on divergence, raise on slow convergence (max 5 retries) |
| `--early-stop` | `False` | stop training when loss is stuck on a high plateau (>0.05, improvement <0.2% over 50 epochs) |
## Output

- `dispx_*.csv` / `dispy_*.csv` — full displacement fields
- `checkpoint.pth.tar` — model checkpoint
- `result_figure.png` — visualisation (ROI colorbar excludes edge pixels)
- `train/` — TensorBoard logs

## Network variants

| Model | Output channels | Matched loss |
|---|---|---|
| `U_DICNet` | 2 (u, v) | `patch_grad` |
| `U_DICNet_shape2` | 12 (u,v + 1st/2nd derivatives) | `patch12` |




## Convergence & reproducibility

- The default scheduler is **ReduceLROnPlateau** . It monitors the loss and automatically halves lr when progress stalls (patience=20).
- Under normal training conditions, U‑DICNet typically converges to a loss on the order of 0.0001 or even lower, often within about 5 minutes for 2,500 epochs on an RTX 4060. However, due to random weight initialisation and variations in the input speckle image pairs, convergence can occasionally stall – and this usually becomes apparent early in training (within the first few dozen epochs), with the loss decreasing very slowly or remaining persistently high.

To address this, we recommend the following steps:

Fix a well‑tested random seed (e.g., --seed 42) to improve stability and reproducibility.
- **`--warmup 200`** — linear lr ramp from 1e-7 to 1e-4 over 200 epochs (verified to converge to ~0.006).
- **`--reverse-lr`** — adaptive lr starting at 1e-6 (verified to converge to ~0.006).
- **`--auto-retry`** — can handle both divergence and slow convergence (5 retries max)
- **`--early-stop`** — stops early when loss is stuck on a high plateau (>0.05), saving time on hopeless runs.
Load a pretrained model (--pretrained) to provide a better starting point.

Manually tune hyperparameters – for example, increase the initial learning rate (--lr 0.005), switch the solver (--solver sgd), These adjustments can often help the model escape local plateaus and resume effective convergence.

