import tensorflow as tf
from tensorflow.python.framework import graph_util
from model import get_embd

import os.path
import sys



def creat_inference_graph():
	images = tf.placeholder(dtype=tf.float32, shape=(None, 112, 112, 3), name='input_images')
	is_training_dropout = tf.constant(False, dtype=bool, shape=[], name='train_phase_dropout')
	is_training_bn = tf.constant(False, dtype=bool, shape=[], name='train_phase_bn')
	embds = get_embd(images, is_training_dropout, is_training_bn)
	embds = tf.identity(embds, 'embeddings')
	return embds


if __name__ == '__main__':
	quantize = True
	start_checkpoint = '/home/fangyu/fy/tflite/fitune_insightface/output/20190924-142833/checkpoints'
	output_names = ['input_images', 'embeddings']
	output_file = '/home/fangyu/fy/tflite/fitune_insightface/output/test.pb'
	# Create the model and load its weights.
	sess = tf.compat.v1.InteractiveSession()
	creat_inference_graph()
	if quantize:
		tf.contrib.quantize.create_eval_graph()
	saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

	for op in tf.get_default_graph().get_operations():
		print(op.name)

	saver.restore(sess, tf.train.latest_checkpoint(start_checkpoint))

	# Turn all the variables into inline constants inside the graph and save it.
	frozen_graph_def = graph_util.convert_variables_to_constants(
		sess, sess.graph_def, output_names)
	tf.io.write_graph(
		frozen_graph_def,
		os.path.dirname(output_file),
		os.path.basename(output_file),
		as_text=False)
	print('Saved frozen graph to %s', output_file)




