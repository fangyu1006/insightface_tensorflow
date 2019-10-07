import os
import tensorflow as tf

graph_def_file = '/home/fangyu/Documents/good_frozen.pb'
#graph_def_file = './output/test.pb'
save_file = './output/converted.tflite'

input_arrays = ["input_images"]
output_arrays = ["embeddings"]

converter = tf.lite.TocoConverter.from_frozen_graph(
        graph_def_file, input_arrays, output_arrays)
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.inference_type = tf.uint8
converter.quantized_input_stats = {input_arrays[0]: (0, 1)}
#converter.default_ranges_stats = (-1,1)
tflite_model = converter.convert()
open(save_file, 'wb').write(tflite_model)

