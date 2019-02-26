import tensorflow as tf

def deconv_layer(x, filter_shape, output_shape, name, strides, padding='SAME', non_linear=tf.nn.elu):
    batch_size = x.shape[0]
    output_shape = tf.stack([tf.shape(x)[0], output_shape[1], output_shape[2], output_shape[3]])
    with tf.variable_scope(name):
        W = tf.get_variable(shape=filter_shape, 
                            name="weight", 
                            initializer=tf.contrib.layers.xavier_initializer(), 
                            dtype=tf.float32)
        b = tf.get_variable(shape=filter_shape[2], 
                            name="bias", 
                            initializer=tf.constant_initializer(0.1), 
                            dtype=tf.float32)
        y = tf.nn.conv2d_transpose(value=x, filter=W, 
                                   output_shape=output_shape, 
                                   strides=strides, padding=padding)
        y = tf.add(y, b)
        
    if non_linear is not None:
        return non_linear(y)
    else:
        return y

def conv_layer(x, filter_shape, name, strides=[1, 1, 1, 1], padding='SAME', non_linear=tf.nn.elu):
    with tf.variable_scope(name):
        W = tf.get_variable(shape=filter_shape, 
                            name="weight", 
                            initializer=tf.contrib.layers.xavier_initializer(), 
                            dtype=tf.float32)
        b = tf.get_variable(shape=filter_shape[3], 
                            name="bias", 
                            initializer=tf.constant_initializer(0.1), 
                            dtype=tf.float32)
    if non_linear is not None:
        return non_linear(tf.add(tf.nn.conv2d(input=x, filter=W, strides=strides, padding=padding), b))
    else:
        return tf.add(tf.nn.conv2d(input=x, filter=W, strides=strides, padding=padding), b)

def linear_layer(x, shape, name, params=None, non_linear=tf.nn.elu):
    # shape: a scalar indicating the number of neurons
    input_shape = x.shape[1]
    with tf.variable_scope(name):
        W = tf.get_variable(shape=[input_shape, shape], 
                            name="weight", 
                            initializer=tf.contrib.layers.xavier_initializer(), 
                            dtype=tf.float32)
        b = tf.get_variable(shape=shape, 
                            name="bias", 
                            initializer=tf.constant_initializer(0.1), 
                            dtype=tf.float32)
    if non_linear is not None:
        return non_linear(tf.add( tf.matmul(x, W), b ))
    else:
        return tf.add( tf.matmul(x, W), b )

def pooling_layer(x, ksize, name, strides, padding='SAME'):
    return tf.nn.max_pool(value=x, ksize=ksize, strides=strides, padding=padding, name=name)

