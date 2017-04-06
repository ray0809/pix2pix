# -*- coding: utf-8 -*-


import numpy as np
import os
from keras.models import Sequential, Model
from keras.layers import Dense, Input, merge
from keras.layers import Reshape,LeakyReLU,ZeroPadding2D
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers.core import Flatten
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD, Adagrad
from PIL import Image
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.objectives import binary_crossentropy
import tensorflow as tf
from tqdm import tqdm
import scipy.misc as im
#K.set_image_dim_ordering('th') 
IN_CH = 3

def convolution(inputs,filters,step,stride=2,Normal=True):
    #use for encoder
    encoder = ZeroPadding2D(padding=(1,1))(inputs)
    encoder = Convolution2D(filters,4,4,subsample=(stride,stride),name='conv_%d'%step)(encoder)
    if Normal:
        encoder = BatchNormalization(name='CBat_%d'%step)(encoder)
    encoder = LeakyReLU(alpha=0.2,name='CLRelu_%d'%step)(encoder)
    return encoder

def deconvolution(inputs,filters,step,dropout):
    _,height,width,_ = (inputs.get_shape()).as_list()
    decoder = Deconvolution2D(filters,4,4,
                              output_shape=(None,2*height,2*width,filters),
                              subsample=(2,2),
                              border_mode='same',
                              name='Deconv_%d' % (8-step))(inputs)
    decoder = BatchNormalization(name='DBat_%d' % (8-step))(decoder)
    if step == 8:
        decoder = Activation(activation='tanh')(decoder)
    else:
        decoder = LeakyReLU(alpha=0.2,name='DLRelu_%d' % (8-step))(decoder)   
    if dropout[step-1] > 0:
        decoder = Dropout(dropout[step-1])(decoder)
    return decoder


def generator_model():
    '''
    global BATCH_SIZE
    # imgs: input: 256x256xch
    # U-Net structure, must change to relu
    g_inputs = Input(shape=(256,256,3))
    encoder_filter = [64,128,256,512,512,512,512]
    Encoder = []
    #Number of encoder or decoder     (same)
    nb_layer = len(encoder_filter)
    #reverse from encoder
    decoder_filter = encoder_filter[::-1]
    dropout = [0.5,0.5,0.5,0,0,0,0,0]
    #Buliding encoder layers...
    for i in range(nb_layer):
        if i == 0:
            encoder = convolution(g_inputs,encoder_filter[i],i+1)
        else:
            encoder = convolution(encoder,encoder_filter[i],i+1)
        Encoder.append(encoder)     
        
    #Middle layer...
    middle = convolution(Encoder[-1],512,8)
    
    #Buliding decoder layers...
    for j in range(nb_layer):
        if j == 0:
            decoder = deconvolution(middle,decoder_filter[j],j+1,dropout)
        else:
            decoder = merge([decoder,Encoder[nb_layer-j]],mode='concat',concat_axis=-1)
            decoder = deconvolution(decoder,decoder_filter[j],j+1,dropout)
            
    #Generate original size's pics
    g_output = merge([decoder,Encoder[0]],mode='concat',concat_axis=-1)
    g_output = deconvolution(g_output,3,8,dropout)
    
    model = Model(g_inputs,g_output)
    return model
    '''
    g_inputs = Input(shape=(256,256,3))
    encoder_filter = [64,128,256,512,512,512,512]
    Encoder = []
    #Number of encoder or decoder     (same)
    nb_layer = len(encoder_filter)
    #reverse from encoder
    decoder_filter = encoder_filter[::-1]
    dropout = [0.5,0.5,0.5,0,0,0,0,0]
    #Buliding encoder layers...
    for i in range(nb_layer):
        if i == 0:
            encoder = convolution(g_inputs,encoder_filter[i],i+1)
        else:
            encoder = convolution(encoder,encoder_filter[i],i+1)
        Encoder.append(encoder)     
        
    #Middle layer...
    middle = convolution(Encoder[-1],512,8)
    
    #Buliding decoder layers...
    for j in range(nb_layer):
        if j == 0:
            decoder = deconvolution(middle,decoder_filter[j],j+1,dropout)
        else:
            decoder = merge([decoder,Encoder[nb_layer-j]],mode='concat',concat_axis=-1)
            decoder = deconvolution(decoder,decoder_filter[j],j+1,dropout)
            
    #Generate original size's pics
    g_output = merge([decoder,Encoder[0]],mode='concat',concat_axis=-1)
    g_output = deconvolution(g_output,3,8,dropout)
    
    model = Model(g_inputs,g_output)
    #model.compile(loss='binary_crossentropy',optimizer='Adam')
    return model

def discriminator_model():
    """ return a (b, 1) logits"""
    '''
    model = Sequential()
    model.add(Convolution2D(64, 4, 4,border_mode='same',input_shape=(img_cols, img_rows,IN_CH*2)))
    model.add(BatchNormalization(mode=2))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 4, 4,border_mode='same'))
    model.add(BatchNormalization(mode=2))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(512, 4, 4,border_mode='same'))
    model.add(BatchNormalization(mode=2))
    model.add(Activation('tanh'))
    model.add(Convolution2D(1, 4, 4,border_mode='same'))
    model.add(BatchNormalization(mode=2))
    model.add(Activation('tanh'))
    
    model.add(Activation('sigmoid'))
    '''
    inputs = Input(shape=(img_cols,img_rows,IN_CH*2))
    d = ZeroPadding2D(padding=(1,1))(inputs)
    d = Convolution2D(64,4,4,subsample=(2,2))(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = ZeroPadding2D(padding=(1,1))(d)
    d = Convolution2D(128,4,4,subsample=(2,2))(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = ZeroPadding2D(padding=(1,1))(d)
    d = Convolution2D(256,4,4,subsample=(2,2))(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = ZeroPadding2D(padding=(1,1))(d)
    d = Convolution2D(512,4,4,subsample=(1,1))(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = ZeroPadding2D(padding=(1,1))(d)
    d = Convolution2D(1,4,4,subsample=(1,1),activation='sigmoid')(d)
    model = Model(inputs,d)
    return model



def generator_containing_discriminator(generator, discriminator):
    inputs = Input((img_cols, img_rows,IN_CH))
    x_generator = generator(inputs)
    
    merged = merge([inputs, x_generator], mode='concat',concat_axis=-1)
    discriminator.trainable = False
    x_discriminator = discriminator(merged)
    
    model = Model(inputs,[x_generator,x_discriminator])
    
    return model

def generate_pic(generator,target,e):
    pic = generator.predict(target)
    pic = np.squeeze(pic,axis=0)
    target = np.squeeze(target,axis=0)
    im.imsave('target_%d.png' % e,target)
    im.imsave('pic_%d.png' % e,pic)

np.mean
def discriminator_on_generator_loss(y_true,y_pred):
    return K.mean(K.binary_crossentropy(y_pred,y_true), axis=(1,2,3))

def generator_l1_loss(y_true,y_pred):
    return K.mean(K.abs(y_pred - y_true),axis=(1,2,3))

def train(epochs,batchsize):
    pic = np.load('pic.npy')
    target = np.load('target.npy')
    pic = pic.astype('float32')
    target = target.astype('float32')
    pic = (pic - 127.5) / 127.5
    target = (target - 127.5) / 127.5
    batchCount = pic.shape[0] / batchsize
    print 'Epochs',epochs
    print 'Bathc_size',batchsize
    print 'Batches per epoch',batchCount
    generator = generator_model()
    discriminator = discriminator_model()
    gan = generator_containing_discriminator(generator,discriminator)
    generator.compile(loss=generator_l1_loss, optimizer='RMSprop')
    gan.compile(loss=[generator_l1_loss,discriminator_on_generator_loss] , optimizer='RMSprop')
    discriminator.trainable = True
    discriminator.compile(loss=discriminator_on_generator_loss, optimizer='RMSprop')
    G_loss = []
    D_loss = []
    for e in xrange(1,epochs+1):
        print '-'*15 , 'Epoch %d' % e , '-'*15
        for _ in tqdm(xrange(batchCount)):
            random_number = np.random.randint(1,pic.shape[0],size=batchsize)
            batch_pic = pic[random_number]
            batch_target = target[random_number]
            batch_target2 = np.tile(batch_target,(2,1,1,1))
            y_dis = np.zeros((2*batchsize,30,30,1))
            y_dis[:batchsize] = 1.0
            generated_pic = generator.predict(batch_target)
            #Default is concat first dimention
            concat_pic = np.concatenate((batch_pic,generated_pic))
            
            dis_input = np.concatenate((concat_pic,batch_target2),axis=-1)
            dloss = discriminator.train_on_batch(dis_input,y_dis)
            random_number = np.random.randint(1,pic.shape[0],size=batchsize)
            train_target = target[random_number]
            batch_pic = pic[random_number]
            y_gener = np.ones((batchsize,30,30,1))
            discriminator.trainable = False
            gloss = gan.train_on_batch(train_target,[batch_pic,y_gener])
            discriminator.trainable = True
        G_loss.append(gloss)
        D_loss.append(dloss)
        if e % 50 == 0 or e == 1:
            generate_pic(generator,target[0:1],e)
            generator.save('Model_para/pix2pix_g_epoch_%d.h5' % e)
            discriminator.save('Model_para/pix2pix_d_epoch_%d.h5' % e)
            gan.save('Model_para/pix2pix_gan_epoch_%d.h5' % e)
    D_loss = np.array(D_loss)
    G_loss = np.array(G_loss)
    np.save('Model_para/dloss.npy',D_loss)
    np.save('Model_para/gloss.npy',G_loss)
            
if __name__ == '__main__':
    train(200,12)
    
    g = generator_model()
    d = discriminator_model()
    gan = generator_containing_discriminator(g,d)
    g.load_weights('Model_para/pix2pix_g_epoch_200.h5')
    d.load_weights('Model_para/pix2pix_d_epoch_200.h5')
    gan.load_weights('Model_para/pix2pix_gan_epoch_200.h5')
    
    
    
