# Unsupervised Deep Learning DIC with Single-pair Image for High Order Displacement Measurement

This repository contains the code implementation for the paper "Unsupervised Deep Learning DIC with only Single-pair image for high order displacement measurement". For a detailed explanation of this code, please refer to the paper.

## Requirements

The code is written in Python 3.7 and uses the following libraries:

- PyTorch 3.7
- ImageIO
- NumPy
- pandas
- matplotlib
- argparse
- tensorboard
- glob
- shutil

## Usage

Before running the code, please change the addresses in the code to the directories containing the reference images and deformed images. The names of the reference images and deformed images should be `re*.bmp` and `tar*.bmp`, respectively.
The radius of the subset used can be modified through the global parameter 'radius', the epoch and learning rate can be modified through parameter "epochs" and "lr"

The code will save the computed displacement field in the same directory, with the filenames `disp_x.csv` and `disp_y.csv`. The parameters after iteration will also be saved.



## Note

The code has not been modified for ease of use. This will be improved in the future. Thank you for your understanding.
