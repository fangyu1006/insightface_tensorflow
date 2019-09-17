import tensorflow as tf

'''
def batch_normalization(input, name, is_train, **kwargs):
    mean = tf.Variable(__weights_dict[name]['mean'], name = name + "_mean", trainable = is_train)
    variance = tf.Variable(__weights_dict[name]['var'], name = name + "_var", trainable = is_train)
    offset = tf.Variable(__weights_dict[name]['bias'], name = name + "_bias", trainable = is_train) if 'bias' in __weights_dict[name] else None
    scale = tf.Variable(__weights_dict[name]['scale'], name = name + "_scale", trainable = is_train) if 'scale' in __weights_dict[name] else None
    return tf.nn.batch_normalization(input, mean, variance, offset, scale, name = name, **kwargs)

def convolution(input, name, group, is_train, **kwargs):
    w = tf.Variable(__weights_dict[name]['weights'], trainable=is_train, name=name + "_weight")
    if group == 1:
        layer = tf.nn.convolution(input, w, name=name, **kwargs)
    else:
        weight_groups = tf.split(w, num_or_size_splits=group, axis=-1)
        xs = tf.split(input, num_or_size_splits=group, axis=-1)
        convolved = [tf.nn.convolution(x, weight, name=name, **kwargs) for
                    (x, weight) in zip(xs, weight_groups)]
        layer = tf.concat(convolved, axis=-1)

    if 'bias' in __weights_dict[name]:
        b = tf.Variable(__weights_dict[name]['bias'], trainable=is_train, name=name + "_bias")
        layer = layer + b
    return layer

def prelu(input, name, is_train):
    gamma = tf.Variable(__weights_dict[name]['gamma'], name=name + "_gamma", trainable=is_train)
    return tf.maximum(0.0, input) + gamma * tf.minimum(0.0, input)
'''
'''
def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck, is_train,**kwargs):
	bn1 = batch_normalization(data, variance_epsilon=2e-05, name=name+'_bn1', is_train=is_train)
	conv1_pad = tf.pad(bn1, paddings = [[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
	conv1 = convolution(conv1_pad, group=1, strides=[1, 1], padding='VALID', name=name+'_conv1', is_train=is_train)
	bn2 = batch_normalization(conv1, variance_epsilon=2e-05, name=name+'_bn2', is_train=is_train)
	relu1 = prelu(bn2, name=name+'_relu1',is_train=is_train)
	conv2_pad = tf.pad(stage1_unit1_relu1, paddings = [[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
	conv2 = convolution(conv2_pad, group=1, strides=stride, padding='VALID', name=name+'_conv2', is_train=is_train)
	bn3 = batch_normalization(conv2, variance_epsilon=2e-05, name=name+'_bn3', is_train=is_train)

	if dim_match:
		shortcut = data
	else:
		conv1sc = convolution(data, group=1, strides=stride, padding='VALID', name=name+'_conv1sc', is_train=is_train)
		shortcut = batch_normalization(conv1sc, variance_epsilon=2e-05, name=name+'_sc', is_train=is_train)
	return bn3 + shortcut
'''
def prelu(input, name):
    gamma = tf.compat.v1.Variable(shape=input.get_shape()[-1], initializer=tf.constant_initializer(0.1), dtype=input.dtype, name=name + "_gamma")
    return tf.maximum(0.0, input) + gamma * tf.minimum(0.0, input)

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck, is_train,**kwargs):
	bn1 = tf.compat.v1.layers.batch_normalization(data, momentum=0.9, epsilon=2e-05, name=name+'_bn1', training=is_train)
	conv1_pad = tf.pad(bn1, paddings = [[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
	conv1 = tf.compat.v1.layers.conv2d(inputs=conv1_pad, filters=num_filter, kernel_size=(3,3),strides=(1,1), padding='valid',use_bias=False,name=name+'_conv1')
	bn2 = tf.compat.v1.layers.batch_normalization(conv1, momentum=0.9, epsilon=2e-05, name=name+'_bn2', training=is_train)
	relu1 = prelu(bn2, name=name+'_relu1')
	conv2_pad = tf.pad(stage1_unit1_relu1, paddings = [[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
	conv2 = tf.compat.v1.layers.conv2d(inputs=conv2_pad, filters=num_filter, kernel_size=(3,3),strides=stride, padding='valid',use_bias=False,name=name+'_conv2')
	bn3 = tf.compat.v1.layers.batch_normalization(conv2, momentum=0.9, epsilon=2e-05, name=name+'_bn3', training=is_train)


	if dim_match:
		shortcut = data
	else:
		conv1sc = tf.compat.v1.layers.conv2d(inputs=data, filters=num_filter, kernel_size=(1,1),strides=stride, padding='valid', use_bias=False,name=name+'_conv1sc')
		shortcut = tf.compat.v1.layers.batch_normalization(conv1sc, momentum=0.9, epsilon=2e-05, name=name+'_sc', training=is_train)
	return bn3 + shortcut

def get_fc1(last_conv, num_classes, is_training_dropout, is_training_bn, input_channel=512):
	body = last_conv
	body = tf.compat.v1.layers.batch_normalization(body, momentum=0.9, epsilon=2e-05, name='bn1'', training=is_training_bn)
	if is_training_dropout:
		body = tf.nn.dropout(body, 0.4)
	fc1 = tf.compat.v1.contrib.layers.flatten(body)
	fc1 = tf.compat.v1.layers.dense(inputs=fc1, units=num_classes, name='pre_fc1', use_bias = True)
	fc1 = tf.compat.v1.layers.batch_normalization(fc1, momentum=0.9, epsilon=2e-05, name='fc1'', training=is_training_bn)
	return fc1


def resnet(inputs, units, num_stages, filter_list, num_classes, bottle_neck, is_training_dropout,is_training_bn):
	num_unit = len(units)
	assert(num_unit == num_stages)

	mulscalar0_second = tf.constant([0.0078125], dtype=tf.float32, name='mulscalar0_second')
	minusscalar0_second = tf.constant([127.5], dtype=tf.float32, name='minusscalar0_second')
	data = inputs - minusscalar0_second
	data = data * mulscalar0_second
	body = data
	conv0_pad = tf.pad(data, paddings = [[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
	conv1 = tf.compat.v1.layers.conv2d(inputs=conv0_pad, filters=filter_list[0], kernel_size=(3,3),strides=(1,1), padding='valid',use_bias=False,name='conv0')
	body = tf.compat.v1.layers.batch_normalization(body, momentum=0.9, epsilon=2e-05, name='bn0', training=is_training_bn)
	body = prelu(body, name='relu0')

	for i in range(num_stages):
		body = residual_unit(data=body, 
							num_filter=filter_list[i+1], 
							stride = (2, 2), 
							dim_match=False,
							name='stage%d_unit%d' % (i + 1, 1), 
							bottle_neck=bottle_neck,
							is_train=is_training_bn,
							**kwargs=**kwargs)
        for j in range(units[i]-1):
        	body = residual_unit(data=body, 
        						num_filter=filter_list[i+1], 
        						stride=(1,1),
        						dim_match=True, 
        						name='stage%d_unit%d' % (i+1, j+2),
        						bottle_neck=bottle_neck, 
        						is_train=is_training_bn,
        						**kwargs=**kwargs)
    fc1 = get_fc1(body, num_classes, is_training_dropout,is_training_bn)
    return fc1






