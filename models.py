from layer import conv_layer, pooling_layer, deconv_layer
import tensorflow as tf
import pdb
    
class UNet(object):
    def __init__(self, x, t, LR, input_shape, output_shape, model_name='U-Net'):
        # optimization setting
        self.LR = LR
        
        # naming setting
        self.model_name = model_name
        
        # model setting
        self.input_shape = input_shape
        self.output_shape = output_shape
        with tf.variable_scope(self.model_name):
            self.x = x
            self.t = t
            self.y = self._forward_pass(self.x)
            #self.x = tf.placeholder(dtype=tf.float32, shape=self.input_shape, name='input')
            #self.t = tf.placeholder(dtype=tf.float32, shape=self.output_shape, name='output')
            #self.y = self._forward_pass(self.x)

    def _forward_pass(self, x):
        # Encoder
        h1 = conv_layer(x, filter_shape=[3, 3, 3, 64], name='L1') # (90, 422, 64)
        h2 = conv_layer(h1, filter_shape=[3, 3, 64, 64], name='L2')
        h3 = pooling_layer(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='L3') # (45, 211, 64)
        h4 = conv_layer(h3, filter_shape=[3, 3, 64, 128], name='L4')
        h5 = conv_layer(h4, filter_shape=[3, 3, 128, 128], name='L5')
        h6 = pooling_layer(h5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='L6') # (23, 106, 128)
        h7 = conv_layer(h6, filter_shape=[3, 3, 128, 256], name='L7')
        h8 = conv_layer(h7, filter_shape=[3, 3, 256, 256], name='L8')
        h9 = pooling_layer(h8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='L9') # (12, 53, 256)
        h10 = conv_layer(h9, filter_shape=[3, 3, 256, 512], name='L10')
        h11 = conv_layer(h10, filter_shape=[3, 3, 512, 512], name='L11')
        # Decoder
        h12 = deconv_layer(h11, filter_shape=[3, 3, 256, 512], strides=[1, 2, 2, 1], output_shape=[-1, 24, 106, 256], name='L12')
        h12 = tf.concat([h12[:, 0:-1, :, :], h8], axis=3)
        h13 = conv_layer(h12, filter_shape=[3, 3, 512, 256], name='L13')
        h14 = conv_layer(h13, filter_shape=[3, 3, 256, 256], name='L14')
        h15 = deconv_layer(h14, filter_shape=[3, 3, 128, 256], strides=[1, 2, 2, 1], output_shape=[-1, 46, 212, 128], name='L15')
        h15 = tf.concat([h15[:, 0:-1, 0:-1, :], h5], axis=3)
        h16 = conv_layer(h15, filter_shape=[3, 3, 256, 128], name='L16')
        h17 = conv_layer(h16, filter_shape=[3, 3, 128, 128], name='L17')
        h18 = deconv_layer(h17, filter_shape=[3, 3, 64, 128], strides=[1, 2, 2, 1], output_shape=[-1, 90, 422, 64], name='L18')
        h18 = tf.concat([h18, h2], axis=3)
        h19 = conv_layer(h18, filter_shape=[3, 3, 128, 64], name='L19')
        h20 = conv_layer(h19, filter_shape=[3, 3, 64, 64], name='L20')
        h21 = conv_layer(h20, filter_shape=[1, 1, 64, self.output_shape[3]], name='L21', non_linear=None)
        return h21
    
    def optimize(self, loss):
        shape = tf.shape(y) # [batch_size, height, width, class]
        self.y = tf.reshape(self.y, [shape[0]*shape[1]*shape[2], shape[3]])
        self.t = tf.reshape(self.t, [shape[0]*shape[1]*shape[2], shape[3]])
        
        self.loss = loss(self.y, self.t)
        self.training = tf.train.AdamOptimizer(self.LR).minimize(self.loss)

