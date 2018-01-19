import os
import tensorflow as tf
from preprocessing import inception_preprocessing,vgg_preprocessing

IMG_SIZE = 299
def _parse_function_train(filename,label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string,channels=3)
    
    #image_transformed= inception_preprocessing.preprocess_image(image_decoded, IMG_SIZE, IMG_SIZE,True,fast_mode=True,add_image_summaries=False)
    #image_transformed= vgg_preprocessing.preprocess_image(image_decoded, IMG_SIZE, IMG_SIZE,True)
    
    image_transformed = tf.image.random_flip_left_right(image_decoded)
    image_transformed = tf.image.resize_images(image_transformed,[IMG_SIZE, IMG_SIZE])
    image_transformed = (image_transformed/255.0 -0.5)*2
    image_transformed.set_shape([IMG_SIZE, IMG_SIZE,3])
    return image_transformed,label

def _parse_function_test(filename,label):  #没有做测试时增强
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string,channels=3)
    #image_transformed= inception_preprocessing.preprocess_image(image_decoded, IMG_SIZE, IMG_SIZE,False,fast_mode=True,add_image_summaries=False)
    #image_transformed= vgg_prerocessing.preprocess_image(image_decoded, IMG_SIZE, IMG_SIZE,False)
    image_transformed = tf.image.resize_images(image_decoded,[IMG_SIZE,IMG_SIZE])
    image_transformed = (image_transformed/255.0 -0.5)*2
    image_transformed.set_shape([IMG_SIZE,IMG_SIZE,3])
    return image_transformed,label

def get_train_dataset(filenames,labels,batch_size,rootdir='data/train'):
    full_paths = [os.path.join(rootdir, item + '.jpg') for item in filenames]
    filenames_tensor = tf.constant(full_paths)
    labels_tensor = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((filenames_tensor, labels_tensor))

    dataset = dataset.shuffle(buffer_size=1000).map(_parse_function_train).batch(batch_size).repeat()
    dataiter = dataset.make_one_shot_iterator()
    x,y = dataiter.get_next()
    return x,y


def get_test_dataset(filenames,batch_size,rootdir='data/test'):
    full_paths = [os.path.join(rootdir, item + '.jpg') for item in filenames]
    filenames_tensor = tf.constant(full_paths)
    filenames_tensor2 = tf.constant(full_paths)
    dataset = tf.data.Dataset.from_tensor_slices((filenames_tensor,filenames_tensor2))

    dataset = dataset.map(_parse_function_test).batch(batch_size).repeat()
    dataiter = dataset.make_one_shot_iterator()
    x,y = dataiter.get_next()
    return x,y

def train_epoch(endpoints,epoch_size,sess,print_every=20):
    total_correct = 0
    total_n_sample = 0
    total_loss = 0
    for batch_idx in range(1,epoch_size+1):
        y,loss,logits,_ = sess.run([endpoints['y'],endpoints['loss'],
                                         endpoints['Logits'],endpoints['optim']])
        cur_n_sample = len(y)
        cur_correct = (logits.argmax(axis=-1)==y).sum()
        cur_loss = loss.sum()
        total_correct += cur_correct
        total_loss += cur_loss
        total_n_sample += cur_n_sample

        if batch_idx % print_every == 0:
            print('current batch: {}/{} \tLoss: {:.6f}\tAcc: {:.6f}'.format(
                batch_idx, epoch_size,
                cur_loss/cur_n_sample,cur_correct/cur_n_sample))
            print('current batch size: ',cur_n_sample)

    accuracy = total_correct / total_n_sample
    total_loss = total_loss / total_n_sample

    print ('Train epoch Acc: {:.6f}\t Loss: {:.6f}'.format(accuracy,total_loss))
    return accuracy

def val_epoch(endpoints,epoch_size,sess):
    total_correct = 0
    total_n_sample = 0
    total_loss = 0
    for batch_idx in range(1,epoch_size+1):
        y,loss,logits= sess.run([endpoints['y'],endpoints['loss'],
                                         endpoints['Logits']])
        cur_n_sample = len(y)
        cur_correct = (logits.argmax(axis=-1)==y).sum()
        cur_loss = loss.sum()
        total_loss += cur_loss
        total_correct += cur_correct
        total_n_sample += cur_n_sample

    total_loss /= total_n_sample
    accuracy = total_correct / total_n_sample
    print('Val epoch Acc {:.6f}\t Loss: {:.6f}.'.format(accuracy,total_loss))
    return accuracy
