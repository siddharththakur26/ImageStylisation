#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import sys
import scipy.io
import scipy.misc
import imageio
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from IPython.display import Image, display


# In[2]:


class Configuration:
    COLOR_CHANNEL = 3
    CONTENT_IMAGE = 'images/content_image.jpg'
    IMAGE_HEIGHT= imageio.imread(CONTENT_IMAGE).shape[0]
    IMAGE_WIDTH= imageio.imread(CONTENT_IMAGE).shape[1]
    STYLE_IMAGE = 'images/style_image.jpg'
    OUPUT = 'images/'
    NOISE_RATIO = 0.6
    ALPHA = 100
    BETA = 5
    PATH_VGG_MODEL = 'pretrained_model/imagenet-vgg-verydeep-19.mat'
    MEAN = np.array([123.68,116.779, 103.939]).reshape((1,1,1,3))


# In[3]:


def load_vgg(path_location):
    vgg = scipy.io.loadmat(path_location)
    vgg_layers = vgg['layers']
    
    def _weights(layer,expected_layer_name):
        weight_basis = vgg_layers[0][layer][0][0][2]
        weights = weight_basis[0][0]
        basises = weight_basis[0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        return weights,basises
    
    def _relu(conv_layer):
        return tf.nn.relu(conv_layer)
    
    def _conv(previous_layer,layer,layer_name):
        wght,bsis = _weights(layer,layer_name)
        wght = tf.constant(wght)
        bsis_reshape = np.reshape(bsis,(bsis.size))  
        bsis = tf.constant(bsis_reshape)
        return tf.nn.conv2d(previous_layer,filter=wght,strides=[1,1,1,1],padding='SAME')+ bsis
    
    def _conv_relu(previous_layer,layer,layer_name):
        return _relu(_conv(previous_layer,layer,layer_name))
    
    def _average_pooling(previous_layer):
        return tf.nn.avg_pool(previous_layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
    # graph model to be implemented
    graph={}
    graph['input']   = tf.Variable(np.zeros((1, Configuration.IMAGE_HEIGHT, Configuration.IMAGE_WIDTH, Configuration.COLOR_CHANNEL)), dtype = 'float32')
    graph['conv1_1']  = _conv_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = _conv_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _average_pooling(graph['conv1_2'])
    graph['conv2_1']  = _conv_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _average_pooling(graph['conv2_2'])
    graph['conv3_1']  = _conv_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _average_pooling(graph['conv3_4'])
    graph['conv4_1']  = _conv_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _average_pooling(graph['conv4_4'])
    graph['conv5_1']  = _conv_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _average_pooling(graph['conv5_4'])
    
    return graph


# In[4]:


def _generate_white_noise_content_image(content_image):
    white_noise_image = np.random.uniform(-20,20,(1,Configuration.IMAGE_HEIGHT,Configuration.IMAGE_WIDTH,Configuration.COLOR_CHANNEL)).astype('float32')
    input_content_image = white_noise_image + content_image
    return input_content_image


# In[5]:


def _reshape_normalize(img):
    img = np.reshape(img,((1,) + img.shape))
    img = img - Configuration.MEAN
    return img


# In[6]:


def _save_image(path_location,output_image):
    output_image = output_image.astype('float32') + Configuration.MEAN
    output_image = np.clip(output_image[0],0,255).astype('uint8')
    imageio.imwrite(path_location,output_image)


# In[7]:


input_c_image = imageio.imread(Configuration.CONTENT_IMAGE)
input_content_image = _reshape_normalize(input_c_image)
input_s_image = imageio.imread(Configuration.STYLE_IMAGE)
input_style_image = _reshape_normalize(input_s_image)



# In[8]:


def _compute_content_loss(activation_content_image, activation_generate_image):
    m, N,M1,M2 = activation_generate_image.get_shape().as_list()
    activation_content_image_reshape = tf.transpose(tf.reshape(activation_content_image,[N*M1,M2]))
    activation_generate_image_reshape = tf.transpose(tf.reshape(activation_generate_image,[N*M1,M2]))
    mean_square_distance = tf.reduce_sum(tf.square(tf.subtract(activation_content_image_reshape,activation_generate_image_reshape)))
    content_loss = (1/(4*N*M1*M2))* mean_square_distance
    return content_loss


# In[9]:


def _gram_matrix(A):
    return tf.matmul(A,tf.transpose(A))


# In[10]:


def _compute_layer_style_loss(activation_style_image,activation_generate_image):
    m,N,M1,M2 = activation_generate_image.get_shape().as_list()
    activation_style_image = tf.transpose(tf.reshape(activation_style_image,[N*M1,M2]))
    activation_generate_image = tf.transpose(tf.reshape(activation_generate_image,[N*M1,M2]))
    gram_matrix_style_image = _gram_matrix(activation_style_image) 
    gram_matrix_generate_image = _gram_matrix(activation_generate_image)
    mean_square_distance = tf.reduce_sum(tf.square(tf.subtract( gram_matrix_generate_image,gram_matrix_style_image)))
    layer_style_loss = (1/(4*M2*M2*(N*M1)*(N*M1))) * mean_square_distance
    return layer_style_loss


# In[11]:


STYLE_IMAGE_LAYER = [('conv1_1',0.2),
                     ('conv2_1',0.2),
                     ('conv3_1',0.2),
                     ('conv4_1',0.2),
                     ('conv5_1',0.2)
                    ]

def _compute_style_loss(model,STYLE_IMAGE_LAYER):
    style_loss = 0
    for layer_name, coeff in STYLE_IMAGE_LAYER:
        ouput_tensor = model[layer_name]
        hidden_activation_layer_style = sess.run(output_tensor)
        hidden_activation_layer_generate = output_tensor
        style_loss = _compute_layer_style_loss(hidden_activation_layer_style,hidden_activation_layer_generate)
    return style_loss

    


# In[12]:


def _total_loss(content_loss,style_loss):
    total_loss = Configuration.ALPHA*content_loss + Configuration.BETA*style_loss
    return content_loss


# In[13]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
content_image = _generate_white_noise_content_image(input_content_image)
model = load_vgg(Configuration.PATH_VGG_MODEL)
sess.run(model['input'].assign(input_content_image))
output_tensor = model['conv4_2']





# In[14]:


hidden_layer_activation_content = sess.run(output_tensor)


# In[15]:


hidden_layer_activation_generate = output_tensor


# In[16]:


content_loss = _compute_content_loss(hidden_layer_activation_content,hidden_layer_activation_generate)


# In[17]:


style_loss = _compute_style_loss(model,STYLE_IMAGE_LAYER)


# In[19]:



total_loss = _total_loss(content_loss,style_loss)


# In[20]:



#setting the training parameters
optimizer = tf.train.AdamOptimizer(2.0)
training_step = optimizer.minimize(total_loss)
path_generate_image = 'images/'+'generate_image'+'.jpg'
iteration = 1
def _model_nn(sess,input_img,path_generate_img,iteration):
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_img))
    
    for i in range(iteration):
        sess.run(training_step)
        generate_image = sess.run(model['input'])
    _save_image(path_generate_image,generate_image)
    return generate_image
   
    


# In[21]:


_model_nn(sess,content_image,path_generate_image,iteration)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




