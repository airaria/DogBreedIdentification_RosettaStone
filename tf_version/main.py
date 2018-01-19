import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 设置可见cuda设备,'0' or '1',若不设置则所有设备可见，若设为''则不可见任何gpu
import sys
sys.path.append('..')
import pandas as pd
from models import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.label import LabelEncoder
from utils import *
import tensorflow as tf
import glob
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.6 #固定使用显存比例
config.gpu_options.allow_growth = True #动态分配显存

BATCH_SIZE = 64
EPOCHS = 10

data_train_csv = pd.read_csv('data/labels.csv')
filenames = data_train_csv.id.values
le = LabelEncoder()
labels = le.fit_transform(data_train_csv.breed)
N_CLASS = len(le.classes_)

filenames_train , filenames_val ,labels_train, labels_val =\
    train_test_split(filenames,labels,test_size=0.1,stratify=labels)
filenames_test = [i.split('/')[-1].split('.')[0] for i in glob.glob('data/test/*')]
EPOCH_TRAIN_SIZE = len(filenames_train)//BATCH_SIZE + 1
EPOCH_VAL_SIZE =   len(filenames_val)//BATCH_SIZE + 1
EPOCH_TEST_SIZE = len(filenames_test)//BATCH_SIZE + 1
sess=tf.Session(config=config)

x_train, y_train = get_train_dataset(filenames_train,labels_train,BATCH_SIZE,rootdir='data/train')
x_val,y_val = get_train_dataset(filenames_val,labels_val,BATCH_SIZE,rootdir='data/train')
#x_test,id_test = get_test_dataset(filenames_test,BATCH_SIZE,rootdir='data/test')

endpoints_train= get_inceptionV3(x_train,y_train,n_class=N_CLASS,reuse=False,is_training=False,mode='dev')
endpoints_val =  get_inceptionV3(x_val,y_val,n_class=N_CLASS,reuse=True,is_training=False,mode='dev')
#endpoints_test = get_resnet50(x_test,id_test,len(le.classes_),reuse=True,is_training=False,mode='test')

restorer = restore_inceptionV3_weights()


learning_rate = tf.get_variable("learning_rate",dtype=tf.float32,initializer=tf.constant(1e-4),trainable=False)
reduce_lr = tf.assign_sub(learning_rate,learning_rate/2)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

training_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'InceptionV3/Logits')
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    mean_loss = tf.reduce_mean(endpoints_train['loss'],name='mean_loss')
    train_op = optimizer.minimize(mean_loss,var_list = training_variables)
endpoints_train['optim'] = train_op

init = tf.global_variables_initializer()
tf.get_default_graph().finalize()

sess.run(init)
restorer(sess)

state = {'val_acc':[],'lives':4,'best_val_acc':0}
print_every = 20


for epoch in range(1,EPOCHS+1):
    print("Epoch: ",epoch)
    train_acc = train_epoch(endpoints_train,EPOCH_TRAIN_SIZE,sess,print_every=20)
    
    print ("Evaluating...")
    val_acc = val_epoch(endpoints_val,EPOCH_VAL_SIZE,sess)

    state['val_acc'].append(val_acc)
    if val_acc > state['best_val_acc']:
        state['lives'] = 4
        state['best_val_acc'] = val_acc
    else:
        state['lives'] -= 1
        print ("Trial left :",state['lives'])
        if state['lives']==2:
            lr = sess.run(reduce_lr)
            print ("New learning rate: ",lr)
        if state['lives']==0:
            break
