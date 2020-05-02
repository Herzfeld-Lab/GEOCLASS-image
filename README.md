# NN_Class Basic Functionality Tutorial

This is meant to be a guide for installing and running some of the basic functionality of the NN Class project. Right now it is only meant to run on MacOS and Linux, and has only been tested on MacOS Catalina and Ubuntu 18.04. At any time, if you run into an problem please submit an issue from the 'Issues' tab with a description of the problem and your operating system. 

## Installing

First, make sure you have both python3, pip and git installed on your machine, this will probably already be the case.

To download the repository, open up a terminal and navigate to the directory you want this project to live. Then, run

```
git clone https://github.com/Herzfeld-Lab/NN_Class.git
```

If you don't have an ssh key for github set up on your machine, it will ask for your github username and password in the terminal. If you want to set up an ssh key to make things easier in the future, you can follow the tutorial [here](https://help.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh)

Once it's finished downloading, navigate into the `NN_Class` directory and run the install dependencies script

```
cd NN_Class
./install_dependencies.sh
```

This might take some time. 

**Note:** I probably missed adding some dependencies to this script. If you run into a dependency problem later on please submit an issue for it and I'll add it

## Getting Test Data

To get the test data for this tutorial, sftp into Snowdon and download `/Volumes/yosemite3/backup/jackh/Data`. Place this Data folder in the `NN_Class` directory. This will probably take a minute to download since the tiff image is large.

## Configuring the test

The way this repository is designed, each individual classification project will have its own folder in `NN_Class/Config`, whic contains everything needed for the project. Namely, this folder contains a `.config` file which defines all the parameters necessary to train and test the neural network, visualize and label split images, tune network hyperparameters and define image preprocessing steps. I've included an example config file for this test run in `NN_Class/Config/mlp_test_negri/mlp_test_negri.config`. Open up this file in a text editor so we can fill out a few lines. Here is a brief description of the configuration parameters:

### Model Parameters:

- `model`:          This defines which Neural Network model to be used. We are using VarioMLP, defined in `VarioMLP.py`
- `num_classes`:    The number of classes to be used. This can be changed and updated as more data is labeled, but we will      start with just 2 clases
- `vario_num_lag`:  
- `hidden_layers`:
- `activations`:

## Creating Dataset

## Labeling Images with Split Tool

## Training

## Testing

## Vizualizing

## Notes
