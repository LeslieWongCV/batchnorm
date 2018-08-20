#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 17:38:48 2018

@author: leslie
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 13:17:50 2018
"""

from __future__ import print_function, division
import h5py  #导入工具包mpfan
import numpy as np
from glob import glob
from keras.models import load_model
import pandas as pd
import pylab
import os
import keras
import time 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'     
#output_path = os.path.join('..','input')        #?
import matplotlib.pyplot as plt
from skimage.util import montage #change montage2d to montage 
#from skimage.util.montage import montage2d      #from skimage.util import montage
#from skimage.color import label2rgb      #unused?
with h5py.File( 'all_patches.hdf5', 'r') as luna_h5:
    all_slices = luna_h5['ct_slices'].value
    all_classes = luna_h5['slice_class'].value  
    print('data', all_slices.shape, 'classes', all_classes.shape)   
    #样本与标签shape:data (6691, 64, 64) classes (6691, 1)
'''
plt.figure()
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
ax1.plot([1, 2], [1, 3])    # 画小图
ax1.set_title('ax1_title')  # 设置小图的标题
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 0))
ax5 = plt.subplot2grid((3, 3), (2, 1))
ax4.scatter([1, 2], [2, 2])
ax4.set_xlabel('ax4_x')
ax4.set_ylabel('ax4_y')
ax3.scatter([1,2],[2,2])
ax3.set_xlabel('ax3_x')
ax3.set_ylabel('ax3_y')    
   ''' 

fig, (ax1, ax2) = plt.subplots(1,2,figsize = (12, 6))   

#ax1 = plt.subplot(121)
#ax2 = plt.subplot(122)

                    #malignant&benign 
plt_args = dict(cmap = 'bone', vmin = -600, vmax = 300)     #???
ax1.imshow(montage(all_slices[np.random.choice(np.where(all_classes>0.5)[0],size = 64)]), **plt_args)
ax1.set_title('Malignant Tiles')
ax2.imshow(montage(all_slices[np.random.choice(np.where(all_classes<0.5)[0],size = 64)]), **plt_args)
ax2.set_title('Benign Tiles')


from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, GlobalAveragePooling2D, \
    warnings, BatchNormalization
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.utils import get_file
from keras.utils import layer_utils

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"


# Modular function for Fire Node

def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    
    return x

# Original SqueezeNet from paper.

def SqueezeNet(input_tensor=None, input_shape=None,
               weights='imagenet',
               classes=1000,
               use_bn_on_input=False,  # to avoid preprocessing
               first_stride=2
               ):
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')                                      #already fixed?

    if input_tensor is None:
        raw_img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if use_bn_on_input:
        img_input = BatchNormalization()(raw_img_input)
    else:
        img_input = raw_img_input

    x = Convolution2D(64, (3, 3), strides=(first_stride, first_stride), padding='valid', name='conv1')(img_input)
    x = Activation('relu', name='relu_conv1')(x)
    x = BatchNormalization()(x)
 
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)
    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = BatchNormalization()(x)

    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x = BatchNormalization()(x)
    x = Dropout(0.5, name='drop9')(x)

    x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)

    out = Activation('softmax', name='loss')(x)#softmax分类函数
#    out = Dense(1, activation='softmax', name='loss')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = raw_img_input

    model = Model(inputs, out, name='squeezenet')

    # load weights
    if weights == 'imagenet':

        weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels.h5',
                                WEIGHTS_PATH,
                                cache_subdir='models')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model

##########################initicalize##################################
import keras                               
lung_node_cnn = SqueezeNet(input_shape = (64, 64, 1),
                           weights = None, classes = 2,
                  use_bn_on_input = True)
# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
#  train the model using RMSprop
lung_node_cnn.compile(loss='categorical_crossentropy',#compile
              optimizer=opt,
              metrics=['accuracy'])
lung_node_cnn.summary() #打印出模型层次概况

##########################split##################################


from keras.utils.np_utils import to_categorical
X_vec = (np.expand_dims(all_slices,-1) - np.mean(all_slices))/np.std(all_slices)   #-mean/std?   to 0-1?
y_vec = to_categorical(all_classes)  
X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec,  
                                                   train_size = 0.75,
                                                   random_state = 1,
                                                   stratify = all_classes)
loss_history = []
    
for i in range(40):
    print('Training %d/_'%i)
        
    loss_history += [lung_node_cnn.fit(X_train, y_train,#输入数据
                                validation_data=(X_test, y_test),#制定验证集
                               shuffle = True,#表示是否在每个epoch前打乱输入样本的顺序
                               batch_size = 100,
                               epochs = 1)]#本函数用以训练模型
    
    

#################################################################
'''
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
X_vec = (np.expand_dims(all_slices,-1) - np.mean(all_slices))/np.std(all_slices)   #-mean/std?   to 0-1?
y_vec = to_categorical(all_classes)  #1 to 2   why transpose?
X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec,  
                                                   train_size = 0.75,
                                                   random_state = 1,
                                                   stratify = all_classes)
'''
#epich = np.cumsum(np.concatenate([np.linspace(1, 1, len(mh.epoch)) for mh in loss_history]))
'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
_ = ax1.plot(epich,
             np.concatenate([mh.history['loss'] for mh in loss_history]), 'b-',
             epich, np.concatenate([mh.history['val_loss'] for mh in loss_history]), 'r-')
ax1.legend(['Training', 'Validation'])
ax1.set_title('Loss')

_ = ax2.plot(epich, np.concatenate([mh.history['acc'] for mh in loss_history]), 'b-',
             epich, np.concatenate([mh.history['val_acc'] for mh in loss_history]),   'r-')
ax2.legend(['Training', 'Validation'])
ax2.set_title('Accuracy')
'''

'''from keras.utils.np_utils import to_categorical
X_vec = (np.expand_dims(all_slices,-1) - np.mean(all_slices))/np.std(all_slices)   #-mean/std?   to 0-1?
y_vec = to_categorical(all_classes)  
for train_index, val_index in kf.split(X_vec):
        loss_history = []
    
        X_train = np.take(X_vec,train_index,axis=0)
        y_train = np.take(y_vec,train_index,axis=0)
        X_test = np.take(X_vec,val_index,axis=0)
        y_test = np.take(y_vec,val_index,axis=0)
    
        lung_node_cnn = SqueezeNet(input_shape = (64, 64, 1),
                           weights = None, classes = 2,
                  use_bn_on_input = True)
# initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

#  train the model using RMSprop
        lung_node_cnn.compile(loss='categorical_crossentropy',#compile
              optimizer=opt,
              metrics=['accuracy'])
        
 #   for i in range(40):
        
        loss_history += [lung_node_cnn.fit(X_train, y_train,#输入数据
                                validation_data=(X_test, y_test),#制定验证集
                               shuffle = True,#表示是否在每个epoch前打乱输入样本的顺序
                               batch_size = 32,
                               epochs = 2)]#本函数用以训练模型
    
 ############################################################

   #     plt.plot(losslist,label=str(lr)+' batch_size:'+str(batch_size))
        plt.plot(np.concatenate([mh.history['acc'] for mh in loss_history]), 'b-',
                         np.concatenate([mh.history['val_acc'] for mh in loss_history]),   'r-',label=str(lr))
        plt.legend(['Training', 'Validation'])
plt.legend()
plt.show'''

   # print('training:%d/10'%i)     



 ##########################train##################################
 
#loss_history = []
#lung_node_cnn = load_model('model_crossval&batch_size=100%nobano.h5')

'''for train_index, val_index in kf.split(X_vec):
    X_train = np.take(X_vec,train_index,axis=0)
    y_train = np.take(y_vec,train_index,axis=0)
    X_test = np.take(X_vec,val_index,axis=0)
    y_test = np.take(y_vec,val_index,axis=0)
    '''
'''for i in range(60):
    
    loss_history += [lung_node_cnn.fit(X_train, y_train,#输入数据
        validation_data=(X_test, y_test),#制定验证集
                               shuffle = True,#表示是否在每个epoch前打乱输入样本的顺序
                               batch_size = 32,
                               epochs = 1)]#本函数用以训练模型
    print('training:%d'%i)     

'''
                            
'''safs= ['1','2']
sum_safs =np.cumsum(conca_safs)
conca_safs = np.concatenate([np.linspace(1,1,len(safs))])
epich_see = np.array(epich)'''



    
 ##########################plot##################################
 #epich =[ range(len(mh.epoch))for mh in loss_history]
'''
epich = np.cumsum(np.concatenate([np.linspace(1, 1, len(mh.epoch)) for mh in loss_history]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
_ = ax1.plot(epich,
             np.concatenate([mh.history['loss'] for mh in loss_history]), 'b-',
             epich, np.concatenate([mh.history['val_loss'] for mh in loss_history]), 'r-')
ax1.legend(['Training', 'Validation'])
ax1.set_title('Loss')

_ = ax2.plot(epich, np.concatenate([mh.history['acc'] for mh in loss_history]), 'b-',
             epich, np.concatenate([mh.history['val_acc'] for mh in loss_history]),   'r-')
ax2.legend(['Training', 'Validation'])
ax2.set_title('Accuracy')

from sklearn.metrics import classification_report
y_pred_proba = lung_node_cnn.predict(X_test)    #predict 
y_pred = np.argmax(y_pred_proba,1)
print('')
print(classification_report(np.argmax(y_test,1),
                      y_pred))
'''

##########################predict##################################

epich = np.cumsum(np.concatenate(
    [np.linspace(1, 1, len(mh.epoch)) for mh in loss_history]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
_ = ax1.plot(epich,
             np.concatenate([mh.history['loss'] for mh in loss_history]),
             'b-',
             epich, np.concatenate(
        [mh.history['val_loss'] for mh in loss_history]), 'r-')
ax1.legend(['Training', 'Validation'])
ax1.set_title('Loss')

_ = ax2.plot(epich, np.concatenate(
    [mh.history['acc'] for mh in loss_history]), 'b-',
                 epich, np.concatenate(
        [mh.history['val_acc'] for mh in loss_history]),
                 'r-')
                 
ax2.legend(['Training', 'Validation'])
ax2.set_title('Accuracy')
plt.savefig('/Users/leslie/助研-wong/pic_acc_loss100NB C.png')

from sklearn.metrics import classification_report
y_pred_proba = lung_node_cnn.predict(X_test)
y_pred = np.argmax(y_pred_proba,1)
print('')
print(classification_report(np.argmax(y_test,1),
                      y_pred))
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(np.argmax(y_test, 1), y_pred_proba[:,1])
pylab.show()
fig, ax1 = plt.subplots(1,1)
ax1.plot(fpr, tpr, 'r-.', label = 'CNN (%2.2f)' % auc(fpr, tpr))
ax1.set_xlabel('False Positive Rate')#假阳性率
ax1.set_ylabel('True Positive Rate')#真阳性率
ax1.plot(fpr, fpr, 'b-', label = 'Random Guess')
ax1.legend()
#plt.savefig('/Users/leslie/助研-wong/pic.png')
plt.savefig('/Users/leslie/助研-wong/pic_ROC100NB C.png')
#y_test_test = np.argmax(y_test,1)

lung_node_cnn.save('model_100NB C.h5')