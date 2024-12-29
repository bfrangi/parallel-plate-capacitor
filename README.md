# Finite Parallel Plate Capacitor Simulation

Code for _Numerical Simulation of a Finite Parallel Plate Capacitor_ developed for an *Electromagnetism and Optics* course of the UC3M *Physics Engineering* degree.

## Contents

This repository contains the code elaborated mainly in March and April of 2022 for an electromagnetism numerical laboratory. The main file in this repo is ```simulation.py```, which holds the code to solve all the questions posed in ```Documents/Lab Guide.pdf```. It uses the initial parameters defined in ```initial_parameters.py```.

The program ```simulation.py``` generates the file ```potential_matrix.npy```, which contains a 3D mesh with the values of the potential around the capacitor. Some other output files include figures and graphs, and they are inclued in the directory ```Plots```.

The lab report is located at ```Documents/Lab Report.pdf```

## Clone this repository
First, make sure you have installed ```git``` on your system (info on how to install [here](https://github.com/git-guides/install-git)). Then, run ```git clone https://github.com/bfrangi/parallel-plate-capacitor.git``` to clone this repository into your current working directory.

## Requirements

To run this code on your system you will need:

- Python version: Python 3 (specifically, this was done using version 3.8.5).

- Manual installation of these python packages: ```matplotlib```, ```numpy``` and ```pytorch```. These can be installed with:

```
pip3 install matplotlib numpy torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
## Run

Run the simulations with:

```
python3 simulation.py
```


 
