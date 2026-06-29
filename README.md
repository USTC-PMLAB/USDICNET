# U-DICNet (refactored)

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

# 12-channel network with patch12 loss (requires U_DICNet_shape2)
python main.py --data-dir ./star_displacement --arch U_DICNet_shape2 --loss patch12
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
| `--seed` | `None` | random seed for reproducibility |
| `--auto-retry` | `False` | enable adaptive lr retry on stalled convergence |

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

## Notes

- This is a refactored version of the original code. All layer parameters, loss coefficients, and numerical behaviour are kept identical.
- The original research code remains untouched in `USDICNET-main/` and `程序/USDICNET/`.

## Convergence & reproducibility

- The default scheduler is **ReduceLROnPlateau** (same as the original paper). It monitors the loss and automatically halves lr when progress stalls (patience=20).
- **Use `--seed` for reproducibility.** Different random initialisations can lead to very different convergence quality —— from ~0.01 to stuck at ~7.4. A verified good seed is `--seed 42`.
- If training does not converge (loss stays high for many epochs), try these steps in order:
  1. **Fix the seed:** `python main.py --seed 42`（verified to reach loss ~0.01）
  2. **Auto-retry:** `python main.py --auto-retry`（auto-detects stalled runs, re-initialises model, doubles lr, up to 3 retries; first 100 epochs must drop >= 80% or retry）
  3. **Try other seeds:** `--seed 123`, `--seed 999`, etc.
  4. **Adjust hyperparameters:** raise lr（`--lr 0.005`）, switch solver（`--solver sgd`）, or increase subset radius（`--radius 3`）
- With a good initialisation, loss converges to ~0.01 after 2500 epochs in ~5 min on an RTX 4060.