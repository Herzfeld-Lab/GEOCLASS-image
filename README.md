# NN_Class User Guide and Documentation

The following is a guide for installing and running the NN Class software, along with a comprehensive documentation of its features. Right now it is only meant to run on MacOS and Linux, and has only been tested on MacOS Catalina and Ubuntu 18.04. At any time, if you run into an problem please submit an issue from the 'Issues' tab with a description of the problem and your operating system. Also, you can message me on the group slack, I'm pretty much always online. Once the initial bugs seemed to be ironed out, I'll keep pushing additional pieces of functionality to this repo until the full software is stable and ready to release (when that time comes).

# Table of Contents

- [Installation](#installation)
  * [Operating System](#operating-system)
  * [Required Packages](#required-packages)
  * [NNClass Installation](#nnclass-installation)
- [Configuration](#configuration)
  * [Setting up Data Folder](#setting-up-data-folder)
  * [Config Parameters](#config-parameters)
    + [Model Parameters](#model-parameters)
    + [Dataset Parameters](#dataset-parameters)
    + [Training Parameters](#training-parameters)
    + [Data Augmentation Parameters](#data-augmentation-parameters)
    + [Visualization Parameters](#visualization-parameters)
- [Datasets](#datasets)
  * [Creating a Dataset](#creating-a-dataset)
  * [Labeling Training Data](#labeling-training-data)
    + [Individual Labeling](#individual-labeling)
    + [Batch Labeling](#batch-labeling)
    + [Visualizing Labels](#visualizing-labels)
    + [Dataset Output](#dataset-output)
- [Training](#training)
  * [Training Options](#training-options)
  * [Training Output](#training-output)
- [Testing](#testing)
  * [Testing Options](#testing-options)
  * [Testing Output](#testing-output)
- [Visualizing](#visualizing)
  * [Loading Test Output](#loading-test-output)
  * [Visualizing Classifications](#visualizing-classifications)
  * [Visualizing Confidence](#visualizing-confidence)
  * [Adding Classifications to Training Data](#adding-classifications-to-training-data)
  * [Saving Classification Figures](#saving-classification-figures)
- [Expanding Functionality(#expanding-functionality)

# Installation
## Operating System
## Required Packages
## NNClass Installation
# Configuration
## Setting up Data Folder
## Config Parameters
### Model Parameters
### Dataset Parameters
### Training Parameters
### Data Augmentation Parameters
### Visualization Parameters
# Datasets
## Creating a Dataset
## Labeling Training Data
### Individual Labeling
### Batch Labeling
### Visualizing Labels
### Dataset Output
# Training
## Training Options
## Training Output
# Testing
## Testing Options
## Testing Output
# Visualizing 
## Loading Test Output
## Visualizing Classifications
## Visualizing Confidence
## Adding Classifications to Training Data
## Saving Classification Figures
# Expanding Functionality
## Adding Classification Models
## Adding Data Augmentation Methods
## Adding Source Data Types






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

The way this repository is designed, each individual classification project will have its own folder in `NN_Class/Config`, whic contains everything needed for the project. Namely, this folder contains a YAML-format `.config` file which defines all the parameters necessary to train and test the neural network, visualize and label split images, tune network hyperparameters and define image preprocessing steps. I've included an example config file for this test run in `NN_Class/Config/mlp_test_negri/mlp_test_negri.config`. Open up this file in a text editor so we can fill out a few lines. Here is a brief description of the configuration parameters:

### Model Parameters:

- `model`:          This defines which Neural Network model to be used. We are using VarioMLP, defined in `VarioMLP.py`
- `num_classes`:    The number of classes to be used. We will start with just 2 classes for this test.
- `vario_num_lag`:  The number of lag values to be used in the directional Variogram during preprocessing
- `hidden_layers`:  The shape of the hidden layers of the MLP network. Detailed description of how this works below
- `activations`:    The activation functions used in the network's hidden layers (right now, only ReLU is implemented)

### Dataset Parameters:

- `img_path`:         The filepath to the geotiff image to be classified.
- `txt_path`:         The filepath to the .npy file containing all the split image data
- `train_path`:       Deprecated - used only to load split images in the old matlab format (file heirarchy with .png)
- `valid_path`:       Deprecated - used only to load split images in the old matlab format (file heirarchy with .png)
- `class_enum`:       A list of class names, of length `num_classes`. 
- `utm_epsg_code`:    EPSG code of the UTM zone the geotiff image is within (33N for Negribreen)
- `split_img_size`:   Size of split images, in pixels (This will be changed to UTM in the next release)
- `train_test_split`: Percentage of images to be kept as training images (0.8 == 80%), the rest are used for testing

### Training Parameters:

- `use_cuda`:       If true, utilizes GPU for training and testing. Requires extra setup 
- `num_epochs`:     Maximum number of epochs to run the training loop
- `learning_rate`:  Initial learning rate for the optimizer
- `batch_size`:     Number of split images to be passed through network before each iteration of the backpropagation
- `optimizer`:      Optimization algorithm to be used during training

### Data Augmentation Parameters:

- `directional_vario`:    Whether to use directional variogram on split images (Always true unless using a CNN model)
- `random_rotate`:        Randomly rotate via variogram before feeding into network
- `random_shift`:         Randomly shift area to perform variogram over (if the split images are not squares)
- `random_contrast`:      Randomly adjust contrast (untested)
- `random_distort`:       Depracated

### Visualization Parameters:

- `contour_path`:   Filepath to list of UTM coordinates of glacier contour (.npy format)
- `bg_img_path`:    Filepath to background image to display visualizations over (depracated, scaled tiff image used instead)
- `bg_img_utm`:     Filepath to list of UTM coordinates of background image (depracated)

For this test run, I have already set up most of the paramaters in `mlp_test_negri.config`. The config file is automatically updated by various scripts to reflect changes such as adding new classes, and creating a split image dataset. To add the tiff image you just downloaded to the config file, paste it's filepath into the `img_path` parameter. This filepath can be absolute, or relative to the `NN_Class` directory, so `Data/WV02_20160625170309/WV02_20160625170309.tif` should be easiest.

## Creating Dataset

Now, it's time to split the geotiff image into a set of smaller split images that can be used for classification. This is one with the `createDatasetFromGeotiff.py` script. This script takes as an argument a `.config` file, and generates a list of split images from the geotiff that fall within the given glacier contour, and do not contain any black background sections. It saves the pixel coordinates and UTM coordinates of each split image, as well as other usefull information such as the affine transform for pixel to UTM transformations in a data structure that is utilized by the training and testing script, and the labeling and visualization tool. To create a split image dataset, run

```
python3 createDatasetFromGeotiff.py Config/mlp_test_negri/mlp_test_negri.config
```

After it's done, open up `mlp_test_negri.config` again. You will see that the `txt_path` parameter has been automatically filled to point to the newly created split image dataset file.

## Labeling Images with Split Tool

Viewing and Labeling split images and classification labels is done with the Split Image Explorer tool. To load our new split image dataset with the tool, run

```
python3 Split_Image_Explorer.py Config/mlp_test_negri/mlp_test_negri.config
```

It will take some time to load up as it loads in the geotiff image and initializes the UI, but when it's done you should see a window like this:

![Split Image Explorer](images/tool1.png)

The right side of the window shows a preview of the geotiff image, with the glacier contour overlaid as well as a crosshairs pinpointing the location of the split image shown on the left side. You can navigate around the geotiff image preview by clicking in the desired location, or using the 'a' and 'd' keys on your keyboard to move one split image at a time.

**Note:** If there's some offset between where you click and where the crosshairs actually move, try minimizing and then maximizing, or maximizing and then minimizing the window, sometimes it boots up to the wrong size initially. I'm working on a fix for it, but I've tested it on both MacOS and Ubuntu and with some finagling it seems to work. If not, submit an issue for it with a screenshot and you OS.

Note the two buttons near the bottom corresponding to the two classes defined in `mlp_test_negri.config`, 'Undisturbed Snow' and 'Other'. Split images can be labeled either by clicking these buttons, or by pressing the number key on your keyboard corresponding to a class (in this case, 0 for Undisturbed Snow and 1 for Other). 

Start off by labeling some undisturbed snow images. You can do this pretty rapidly from where the crosshair initializes by just pressing 0-d-0-d-0-d-0-d over and over again. Then, click into a few areas of the image preview that are not undisturbed snow, and repeat the process (1-d-1-d-1-d-1-d...). It should only take a minute or two until you've labeled about 100 of each. After labeling, toggle the 'Visualize Labels' checkbox to see the images you've labeled. After labeling a bunch of Undisturbed Snow in the top corner, and a few stripes of Other in the crevassed areas this is what my preview looked like:

![A few labeled images](images/tool2.png)

Here dark blue and light blue correspond to Undisturbed Snow and Other respectively (I will be fleshing out the color coding in a future commit). To save your labels, simply close the Split Image Explorer window. 

## Training

Now that we have some labels, it's time to do a basic training run with our Neural Network model. The `train.py` script takes as an argument a `.config` file and takes care of the whole process. 

```
python3 train.py Config/mlp_test_negri/mlp_test_negri.config
```

After some time loading, you will begin to see printouts of the training and validation loss after each epoch.

**Note:** Right now this training script is extremely memory-intensive. I'm working on a fix that will roughly cut the memory usage in half, but if you have an older computer that only has like 8GB of RAM it's going to slow it way down, especially if there's a lot of other stuff open.

You can stop the training whenever you feel like it using crtl+c in the terminal, or let it run the full 100 epochs. Either way, checkpoints will be saved when the validation loss reaches a new minimum. Once the training has been ended, check the `NN_Class/Output` directory. You will see a folder named with a timestamp for when you started the training session. Within this folder, there is a `checkpoints`, `labels` and `losses` directory. In the `checkpoints` directory you will see the checkpoints that have been saved as your network was training. The most recent of these will be the checkpoint for which the validation loss was lowest.

## Testing

After training, any neural network checkpoint can be used to run a test classification across the whole dataset. In order to do this, the test script takes as an argument a `.config` file, and a checkpoint file specified with the `--load_checkpoint` parameter. For me, the command looked like this

```
python3 test.py Config/mlp_test_negri/mlp_test_negri.config --load_checkpoint Output/01-05-2020_23\:26/checkpoints/epoch_57
```

But the argument for `--load_checkpoint` will be different for you. When complete, the test script generates labels based on the trained model checkpoint in the `Output/[your timestamp here]/labels` directory. The file generated from my command above was `Output/01-05-2020_23\:26/labels/labeled_epoch_57.npy`

## Vizualizing

We can visualize the labels predicted by the neural network using the Split Image Explorer tool. To view labels generated by a training checkpoint run the tool as before, with the additional `--load_labels` argument. For me, this was

```
python3 Split_Image_Explorer.py Config/mlp_test_negri/mlp_test_negri.config --load_labels Output/02-05-2020_13\:17/labels/labeled_epoch_57.npy
```

Using the Visualize Labels checkbox, the labels can be seen. Here is what mine looked like:

![Classification from NN](images/tool3.png)

The slider represents a minimum confidence threshold on a scale from 0% to 100% for visualizing labels. When the Visualize Labels checkbox is toggled, only split images with a confidence greater than the value set by the slider are shown. 

**Note:** The Visualize Checkbox must be un toggled and re toggled any time a change is made in order for the preview to update.

If you want to add an additional class, simply type the class name in the text box under the slider and press enter. The new class will be created, and a new button and checkbox will be generated for it. It will also be mapped to the corresponding number key for labeling. When the Split Image tool is closed, these changes will be reflected in the `num_classes` and `class_enum` parameters in the config file, so retraining the network with new classes remains extremely simple. 

## Next

If you have the time, try playing around with parameters in the config file such as the batch size and hidden layers of the network. Try creating new classes and retraining, and visualizing the results. Please try to break whatever you can, and submit an issue of what you broke and how so I can address bugs.
