# CUMULO Project 

This repository was created during the practical course "Creation of Deep Learning Methods" in the winter semester 2021. 
Its purpose was to use equivariant deep learning architectures to predict cloud types using the CUMULO dataset.

The repository is based on https://github.com/FrontierDevelopmentLab/CUMULO. The U-Net architecture and training script 
were adapted from https://github.com/LobellLab/weakly_supervised.

## Installation

```
cd cumulo
pip install -e .
```

The requirements.txt contains all python 3.8.5 libraries from a standard envrionment on ubuntu 20.04 together with the
essential libraries for this project:

```
absl-py==0.12.0
e2cnn==0.1.7
netCDF4==1.5.1.2
torch==1.5.0
tqdm==4.60.0
```

## Demo

```
mkdir -p /tmp/models
cd cumulo
python scripts/pipeline/train.py --m_path /tmp/models/demo_training --d_path demo_data/ --model unet --epoch_number 5
python scripts/pipeline/predict.py --flagfile /tmp/models/demo_training/flagfile.txt --output_path /tmp/models/demo_training/predictions --prediction_number 50
```


## Structure and Examples

All examples assume that this repository is the current folder.

#### scripts

+ #### preprocessing

    + **calculate_statistics.py**: Can be used to calculate mean, variance and class weights for a given dataset. Example:
    `python scripts/calculate_statistics.py --path <data_path> --sample_number 1000 --tile_number 16` to use 16 tiles of 
    1000 nc files for calculating the statistic.  

    + **filter_artefacts_and_nolabels.py**: The CUMULO dataset contains nc files where missing data was filled up 
    with the nearest available data. This script can be used to filter these artefacts (vertical lines with equal 
    values). There also exist nc files without labels. This script creates a list of these files which can be used 
    to exclude them from training. Example:`python scripts/filter_artefacts_and_nolabels.py --path <data_path>
    --removed_path <other_path>` to sort nc files with artefacts into 'other_path' and to save a list of nc files
    without labels in 'data_path'


+ #### pipeline

    + **train.py**: Can be used to train a model on the CUMULO dataset. All training parameters are given and saved
    as flags from the abseil library. Example: `python scripts/pipeline/train.py --m_path <train_path> --d_path <data_path> 
    --model equi --rot 4` to train an equivariant U-Net (4 discrete rotations) on the data at 'data_path' and save 
    the model and all training information at 'train_path'. Training with the iresnet model is still work-in-progress 
    and throws an error in evaluation and prediction.
    
    + **predict.py**: Can be used to generate predictions using a trained model and to evaluate these predictions,
    generating images, ROCs, histograms, evaluation reports and confusion matrices. Example: 
    `python scripts/pipeline/predict.py --flagfile <path_to_flagfile.txt> --output_path 
    <output_path> --prediction_number 50` to generate and evaluate predictions of 50 nc files from the test set 
    of the trained model which is described by the flagfile.txt in its training folder and save all results at 
    'output_path'
    

+ #### visualization

    + **dataset.py**: Can be used to convert nc files to images.
    + **metrics.py**: Can be used to visualize metrics which are saved during training of models.
    + **models_to_table.py**: Can be used to merge the results of multiple training evaluations into a single 
    LaTeX table.
    + **predictions.py**: Can be used to visualize .npz predictions which have been saved by the predict.py script.
    + **tiles.py**: Can be used to visualize training examples as they are saved by the train.py script.
    

#### cumulo

+ #### data
    + Contains the CumuloDataset which is used to receive tiles during training. 

+ #### models
    + **iresnet.py**: Copied from https://github.com/FrontierDevelopmentLab/CUMULO. For details, see 
    https://arxiv.org/abs/1902.02767.
    + **unet.py**: Adapted U-Net implementation from https://github.com/LobellLab/weakly_supervised.
    + **unet_equi.py**: Same architecture as `unet.py`, but with equivariant modules from the e2cnn 
    library.

+ #### utils
    + **basics.py**: Simple functions for reading nc files, processing labels, ...
    + **evaluation.py**: Functions for performing the evaluation tasks (generating histograms,
    ROCs, evaluation reports, ...).
    + **iresnet_utils.py**: Functions necessary for the iresnet, all copied from 
    https://github.com/FrontierDevelopmentLab/CUMULO. This is still work-in-progress.
    + **training.py**: Functions used during training or by the CumuloDataset (e.g. tile extraction,
    dataset statistics, ...)
    + **visualization.py**: Functions used to convert network predictions into RGB images.
    
    
## Future Work

As indicated, the IResNet model integration into the trainings pipeline is still work-in-progress. The
UNets produce worse results than the IResNet results from https://arxiv.org/abs/1911.04227. Therefore,
the next steps would be to reproduce the IResNet results and understand why it is better in learning
the cloud classes. Vladimir suggested that implementing an equivariant version of the IResNet may also
improve the results. 

The equivariant UNet architecture could still be extended with equivariance to mirroring.