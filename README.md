# Group15_Project21_-02456
This repository contains the code used to reproduce the main results presented in the Group 15 Project 21 report for 02456 Deep Learning (Fall 2025) at DTU.

## Environment Setup 

"UNet" folder is our core codebase. It includes the model architecture, training functions, and other utility methods that support the entire project.

Since it is a submodule, you need to run the following command the first time to update the code:

``` bash
git submodule update --init --recursive
cd unet
pip install -r requirements.txt
```