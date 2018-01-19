import numpy as np
import pandas as pd
import torch
from models import get_resnet50
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.label import LabelEncoder
from torch import nn,optim

from utils import *

BATCH_SIZE =  64
EPOCHS = 80
CUDA = torch.cuda.is_available()

data_train_csv = pd.read_csv('data/labels.csv')
filenames = data_train_csv.id.values
le = LabelEncoder()
labels = le.fit_transform(data_train_csv.breed)

filenames_train , filenames_val ,labels_train, labels_val =\
    train_test_split(filenames,labels,test_size=0.1,stratify=labels)

dog_train = get_train_dataset(filenames_train,labels_train,BATCH_SIZE,rootdir='data/train')
dog_val = get_train_dataset(filenames_val,labels_val,BATCH_SIZE,rootdir='data/train')

net = get_resnet50(n_class = len(le.classes_))
criterion_train = nn.CrossEntropyLoss()
criterion_val = nn.CrossEntropyLoss(size_average=False)

optimizer = optim.Adam(net.fc.parameters(),lr=0.0001) #use default learning rate

#optimizer = optim.SGD(net.fc.parameters(), lr=0.001, momentum=0.9)
#exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

state = {'val_acc':[],'lives':4,'best_val_acc':0}

if CUDA:
    net.cuda()
for epoch in range(1,EPOCHS+1):
    print("Epoch: ",epoch)
    train_acc = train_epoch(net,dog_train,criterion_train,optimizer,CUDA)
    print ("Evaluating...")
    val_acc = val_epoch(net,dog_val,criterion_val,CUDA)

    state['val_acc'].append(val_acc)
    if val_acc > state['best_val_acc']:
        state['lives'] = 4
        state['best_val_acc'] = val_acc
    else:
        state['lives'] -= 1
        print ("Trial left :",state['lives'])
        if state['lives']==2:
            optimizer.param_groups[0]['lr']/=2
        if state['lives']==0:
            break

'''
for param in net.parameters():
    param.requires_grad = True
optimizer = optim.Adam(net.fc.parameters(), lr=0.0001)
state = {'val_acc':[],'lives':3,'best_val_acc':0}
for epoch in range(1,EPOCHS+1):
    print("Epoch: ",epoch)
    train_acc = train_epoch(net,dog_train,criterion_train,optimizer,CUDA)
    print ("Evaluating...")
    val_acc = val_epoch(net,dog_val,criterion_val,CUDA)

    state['val_acc'].append(val_acc)
    if val_acc > state['best_val_acc']:
        state['lives'] = 3
        state['best_val_acc'] = val_acc
    else:
        state['lives'] -= 1
        print ("Trial left :",state['lives'])
        if state['lives']==0:
            break

'''
