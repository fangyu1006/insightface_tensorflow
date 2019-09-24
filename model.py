import tensorflow as tf
from backbones import fresnet

def get_embd(inputs, is_training_dropout, is_training_bn, reuse=False, scope='embd_extractor'):
    num_classes = 512
    num_layers = 18
    filter_list = [64,64,128,256,512]
    bottle_neck = False
    num_stages = 4

    if num_layers == 18:
        units = [2, 2, 2, 2]
    elif num_layers == 34:
        units = [3, 4, 6, 3]
    elif num_layers == 49:
        units = [3, 4, 14, 3]
    elif num_layers == 50:
        units = [3, 4, 14, 3]
    elif num_layers == 74:
        units = [3, 6, 24, 3]
    elif num_layers == 90:
        units = [3, 8, 30, 3]
    elif num_layers == 98:
        units = [3, 4, 38, 3]
    elif num_layers == 99:
        units = [3, 8, 35, 3]
    elif num_layers == 100:
        units = [3, 13, 30, 3]
    elif num_layers == 134:
        units = [3, 10, 50, 3]
    elif num_layers == 136:
        units = [3, 13, 48, 3]
    elif num_layers == 140:
        units = [3, 15, 48, 3]
    elif num_layers == 124:
        units = [3, 13, 40, 5]
    elif num_layers == 160:
        units = [3, 24, 49, 3]
    elif num_layers == 101:
        units = [3, 4, 23, 3]
    elif num_layers == 152:
        units = [3, 8, 36, 3]
    elif num_layers == 200:
        units = [3, 24, 36, 3]
    elif num_layers == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    net = fresnet.resnet(inputs  = inputs,
    			units       = units,
        		num_stages  = num_stages,
        		filter_list = filter_list,
        		num_classes = num_classes,
        		bottle_neck = bottle_neck,
        		is_training_dropout = is_training_dropout,
        		is_training_bn = is_training_bn)

    return net

	
