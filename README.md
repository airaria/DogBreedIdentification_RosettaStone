# Dog Breed Identification
Using pretrained models in tensorflow and pytorch to identify dog breeds.

## Purpose of the repo
is to show how to 
- load and fine-tune pretrained model in tensorflow and pytorch,
- read large dataset using the **dataset** paradigm in tensorflow and pytorch,
- do data augmentation on image data,
- make a comparision between tensorflow and pytorch.

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
The average accuracies of both models reach 90% on val set after 10 training epochs.

## Acknowledgments

tf_version/preprocessing and tf_version/nets are borrowed from slim :https://github.com/tensorflow/models/tree/master/research/slim
