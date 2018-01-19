import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import resnet_v1, inception_v3


def get_inceptionV3(x,y,n_class,reuse,is_training,mode):
    arg_scope = inception_v3.inception_v3_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = inception_v3.inception_v3(x,
                                                    is_training=is_training,
                                                    num_classes=n_class,
                                                    reuse=reuse,scope='InceptionV3',global_pool=False)
    assert logits == end_points['Logits']
    if mode=='dev':
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    else:
        loss = None

    end_points['loss'] = loss
    end_points['y'] = y
    return end_points

def restore_inceptionV3_weights(ckpt_fn='./tf_version/inception_v3.ckpt',full=False):
    if full:
        restore_variables = slim.get_variables_to_restore()
    else:
        restore_variables = slim.get_variables_to_restore(exclude=['InceptionV3/Logits', 'InceptionV3/AuxLogits'])
    restorer = tf.contrib.slim.assign_from_checkpoint_fn(ckpt_fn,restore_variables)
    return restorer


'''
restore_variables = [v for v in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES) if
                     'InceptionV3/Logits' not in v.name and 'InceptionV3/AuxLogits' not in v.name]
restorer = slim.assign_from_checkpoint_fn('./tf_version/inception_v3.ckpt',restore_variables)
restorer = tf.train.Saver(restore_variables)
restorer.restore(sess,'./tf_version/resnet_v2_50.ckpt')
'''


def get_resnet50(x,y,n_class,reuse,is_training,mode):
    arg_scope = resnet_v1.resnet_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = resnet_v1.resnet_v1_50(x,is_training=is_training,
                                                    num_classes=n_class,
                                                    reuse=reuse,spatial_squeeze=True,global_pool=True)
    end_points['Predictions'] = end_points['predictions']
    end_points['Logits'] = logits

    if mode=='dev':
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_points['Logits'], labels=y)
    else:
        loss = None

    end_points['loss'] = loss
    end_points['y'] = y
    return end_points

def restore_resnet_v1_50_weights(ckpt_fn='./tf_version/resnet_v1_50.ckpt',full=False):
    if full:
        restore_variables = slim.get_variables_to_restore()
    else:
        restore_variables = slim.get_variables_to_restore(exclude=['resnet_v1_50/logits'])
    restorer = tf.contrib.slim.assign_from_checkpoint_fn(ckpt_fn,restore_variables)
    return restorer
