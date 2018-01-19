#Dog Breed Identification
Using pretrained models in tensorflow and pytorch to identify dog breeds.

## Purpose of the repo
is to show how to 
1. load and fine-tune pretrained model in tensorflow and pytorch,
2. read large dataset using the **dataset** paradigm in tensorflow and pytorch,
3. do data augmentation on image data,
4. make a comparision between tensorflow and pytorch.

##Usage
1.Download the data from kaggle: www.kaggle.com/c/dog-breed-identification .

2.Unzip labels.csv.zip and train.zip and put them under data/ folder,  now the data/ folder contains labels.csv and a new folder train/  .

3.Train pytorch model:

​	run 

```shell
python pytorch_version/main.py
```



4.Or train tensorflow model:

​	Download inceptionV3 model file from https://github.com/tensorflow/models/tree/master/research/slim , unzip, and put it under tf_version/ , run:

```bash
python tf_version/main.py
```



## Acknowledgments

tf_version/preprocessing and tf_version/nets are borrowed from slim :https://github.com/tensorflow/models/tree/master/research/slim