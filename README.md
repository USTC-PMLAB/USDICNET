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

## pretrained model
The pretrained model of U_DICNet are avaliable at [google drive](https://drive.google.com/file/d/1vtCL7nBXYUPYgWmKGc2iuZGEb4zFY5md/view?usp=share_link) and [百度云盘](https://pan.baidu.com/s/1N99rpZ7-OOgSm6SAvOUo9A?pwd=76tk)


## Convergence & reproducibility

- The default scheduler is **ReduceLROnPlateau** . It monitors the loss and automatically halves lr when progress stalls (patience=20).
- Under normal training conditions, U‑DICNet typically converges to a loss on the order of 0.01 or even lower, often within about 5 minutes for 2,500 epochs on an RTX 4060. However, due to random weight initialisation and variations in the input speckle image pairs, convergence can occasionally stall – and this usually becomes apparent early in training (within the first few dozen epochs), with the loss decreasing very slowly or remaining persistently high.

To address this, we recommend the following steps (try them in order):

Fix a well‑tested random seed (e.g., --seed 42) to improve stability and reproducibility.

Load a pretrained model (--pretrained) to provide a better starting point.

Enable the auto‑retry mechanism (--auto-retry), which automatically detects stagnation, re‑initialises the model, and adjusts the learning rate.

Manually tune hyperparameters – for example, increase the initial learning rate (--lr 0.005), switch the solver (--solver sgd), or enlarge the subset radius (--radius 3). These adjustments can often help the model escape local plateaus and resume effective convergence.

