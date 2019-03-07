from models import UNet
from utils import entropy_loss
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import time

x_train_path = './Dataset/sample_train_color/'
t_train_path = './Dataset/sample_train_label/'
x_train_name = os.listdir(x_train_path)
t_train_name = os.listdir(t_train_path)
x_train_name = [x_train_path+s for s in x_train_name]
x_train_name = x_train_name[0:100]
x_train_name.sort()
t_train_name = [t_train_path+s for s in t_train_name]
t_train_name = t_train_name[0:100]
t_train_name.sort()

# parameters
batch_size = 32
epoch      = 100
LR         = 1e-5
img_height = 90
img_width  = 422
down_scale = 8
class_num  = 9
data_size  = len(x_train_name)
n_batches  = int(math.ceil(data_size/batch_size))


# This cell is used to construct the pipeline of dataset
def _parse_function(x_name, t_name, img_shape, down_scale):
    x_string = tf.read_file(x_name)
    x = tf.image.decode_jpeg(x_string, channels=3)
    x = x[1560:2280, 7:-7]/1000
    x = tf.image.resize_images(x, img_shape)
    t_string = tf.read_file(t_name)
    t = tf.image.decode_png(t_string, channels=1, dtype=tf.uint16)
    t = t[1560:2280, 7:-7]
    t = t[::down_scale, ::down_scale]
    t = tf.cast(t/1000, tf.int32)
    
    shape = tf.shape(t)
    t = tf.reshape(t, (shape[0]*shape[1],))
    t = tf.one_hot(t, depth=41)
    t = tf.concat([t[:, 0:1], t[:, 33:]], axis=1)
    t = tf.reshape(t, (shape[0], shape[1], 9))
    
    return x, t

x_filenames = tf.constant(x_train_name)
t_filenames = tf.constant(t_train_name)

dataset = tf.data.Dataset.from_tensor_slices((x_filenames, t_filenames))
dataset = dataset.map(lambda x, y: _parse_function(x, y, (img_height, img_width), down_scale))
dataset = dataset.shuffle(buffer_size=32).batch(batch_size).repeat(epoch+1)
iterator = dataset.make_initializable_iterator()
next_batch = iterator.get_next()

x_batch, t_batch = next_batch # get the tf variable of input and target images


unet = UNet(x=x_batch, t=t_batch,
            LR=1e-8, input_shape=[None, img_height, img_width, 3], 
            output_shape=[None, img_height, img_width, class_num], )
unet.optimize(entropy_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(iterator.initializer)


saver = tf.train.Saver(max_to_keep=epoch)
if not os.path.isdir('./Models'):
    os.mkdir('./Models')
    os.mkdir('./Models/U-Net/')
elif not os.path.isdir('./Models/U-Net/'):
    os.mkdir('./Models/U-Net/')

total_loss = 0
for _ in range(n_batches):
    loss = sess.run([unet.loss])
    total_loss += loss
    end = time.time()
    message = 'Epoch: {:>2} | Loss: {:>10.8f} | Time: {:>6.1f}'
    print(message.format(ep+1, total_loss/n_batches, end-start))
    
    if os.path.isfile('./log'):
        os.remove('./log')
    with open('./log', 'a') as file_write:
        file_write.write(message.format(0, total_loss/n_batches, end-start))
        file_write.write('\n')

for ep in range(epoch):
    total_loss = 0
    start = time.time()
    for _ in range(n_batches):
        _, loss = sess.run([unet.training, unet.loss])
        total_loss += loss
    end = time.time()
    message = 'Epoch: {:>2} | Loss: {:>10.8f} | Time: {:>6.1f}'
    print(message.format(ep+1, total_loss/n_batches, end-start))
    
    if not os.path.isdir('./Models/U-Net/unet-'+str(ep)):
        os.mkdir('./Models/U-Net/unet-'+str(ep))
    save_path = saver.save(sess, './Models/U-Net/unet-'+str(ep)+'/unet.ckpt')
    
    with open('./log', 'a') as file_write:
        file_write.write(message.format(ep+1, total_loss/n_batches, end-start))
        file_write.write('\n')
