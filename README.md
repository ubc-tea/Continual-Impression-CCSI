# CCSI

This is the PyTorch implemention of our paper **"CCSI: Continual Class-Specific Impression for Data-free Class
Incremental Learning"** accepted to TMI, as an extention work of [**"Class Impression for Data-Free Incremental
Learning"**](https://link.springer.com/chapter/10.1007/978-3-031-16440-8_31) accpeted by MICCAI 2022
by [Sana Ayromlou](https://github.com/sanaAyrml), Tresa Tsang, Prang Abolmaesumi
,and [Xiaoxiao Li](https://xxlya.github.io/xiaoxiao/)

## Abstract

> Our contributions are summarised as:
> - We propose **CCSI** to generate prototypical synthetic images with high quality for each class by 1) initializing
    the synthesis process with the mean image of each class, and 2) regularizing pixel-wise optimization loss of image
    synthesis using moments of normalization layers.
> - We leverage CN in our pipeline and design a novel image synthesis method with class-adaptive CN statistics to 1)
    improve the quality and class-specificity of **CCSI**, and 2) reduce overwriting the moments of normalization layer
    for samples of newly introduced classes to alleviate catastrophic forgetting of previous classes.
> - We introduce several novel losses to mitigate domain shift between synthesized images and original images, handle
    class imbalance issues and encourage robust decision boundaries for handling catastrophic forgetting.

![avatar](./images/main_image.png)
Two main steps of **CCSI** contain: 1) Continual class-specific data synthesis: Initialize a batch of images with the
mean of each class to synthesize images using a frozen model trained on the previous task. Update the batch by
back-propagating and using the moments saved in the CN as a regularization term; 2) Model update on new tasks: Leverage
information from the previous model using the distillation loss. Overcome data imbalance and prevent catastrophic
forgetting of past tasks with the cosine normalized cross-entropy (CN-CE) loss and margin loss. Mitigate domain shift
between synthesized and original data with a novel intra-domain conservative (IdC) loss, a semi-supervised domain
adaptation technique.

## Sections

The main folder contains the code implemented for the MedMNIST dataset. You can get access to MedMNIST dataset
via [Zendo](https://doi.org/10.5281/zenodo.6496656). You could also use our code to download automatically by
setting `download_data=True` in [config](configs) files.

### Dataloader
Contains dataloader implemented for each of datasets. You can add your own dataloader or adjust the implementation of other dataloaders. Make sure to import your dataloader in [main.py](main.py).
### Model
Contains modified Resnet model architecture used in CCSI. Folder [models/layers](models/layers) contains implementation of introduced two novel layers **"Cosine Linear and Continual Normalization"**.
### Wandb
We use Wandb sweep to plot our results and hyperparameter tuning. Replace project name and your wandb key in `wandb_acc` and `wandb_key` accordingly in [config](configs) files.
### Config
All hyperparamteres are set in [config](configs) as `.yaml` files for each dataset. Each task has a separate confid and hyperparameters which can be adjusted.
### Training Scheme
The class incremental training procedures, loss functions and ... are implemented in [incremental_train_and_eval.py](incremental_train_and_eval.py).
### Synthesis Scheme
The continual class specific impression synthesis and its loss functions are implemented in [continual_class_specific_impression.py](data_synthesis/continual_class_specific_impression.py).
## Install
### Requirements

Requirements can be installed using:
```
pip install -r requirements.txt
```
### Running the Code
In order to run the code you use configs provided for each task of each dataset. To run each config:

```
wandb sweep configs/[dataset_name]/sweep_task_[task_number].yaml
```
Your trained models will be saved in `../saved_models/[dataset_name]/[model_name]`. You need to put this address in `saved_model_address` in each config to keep training for following tasks use the same command as above.

## Acknowledgement

The code is includes borrowed implementations from (https://github.com/hshustc/CVPR19_Incremental_Learning)
and (https://github.com/NVlabs/DeepInversion). Thanks for their great works.

## Citing

```
Will be added
```

## Contact

For questions and suggestions, please contact Sana Ayromlou via ayromlous@gmail.com.
