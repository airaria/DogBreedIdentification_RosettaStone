# Dog Breed Identification
Fine-tune pretrained models in tensorflow and pytorch to identify dog breeds, mostly for pedagogical purpose. 

## Purpose of the repo
is to show how to 
- load and fine-tune pretrained model in tensorflow and pytorch,
- read large dataset using the **dataset** paradigm in tensorflow and pytorch,
- do data augmentation on image data in tensorflow and pytorch,
- make a comparision between tensorflow and pytorch.

## Prerequisites

- tensorflow >= 1.3
- pytorch >= 0.3
- numpy, pandas, sklearn, Pillow

## Usage
1. Download the data from kaggle: www.kaggle.com/c/dog-breed-identification 

2. Unzip labels.csv.zip and train.zip and put them under data/ folder,  now the data/ folder contains labels.csv and a new folder train/  .

3. To train pytorch model, run

```shell
python pytorch_version/main.py
```

4. To train tensorflow model, download inceptionV3 model file from https://github.com/tensorflow/models/tree/master/research/slim , unzip, and put it under tf_version/, run:

```shell
python tf_version/main.py
```

## Results
After 10 training epochs:
- tensorflow model (using InceptionV3): mean accuracy reaches 90% on val set.
- pytorch model (using Resnet50): mean accuracy reaches 87% on val set.

## Acknowledgments

tf_version/preprocessing and tf_version/nets are borrowed from slim :https://github.com/tensorflow/models/tree/master/research/slim
