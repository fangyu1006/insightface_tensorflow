import numpy as np
import tensorflow as tf


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/home/fangyu/fy/tflite/fitune_insightface/output/test_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

image = np.random.rand(1,112,112,3)
#image = np.ones((1,112,112,3))
#image = (image-128)*0.007843137718737125
image_ = image.astype('float32')
interpreter.set_tensor(input_details[0]['index'], image_)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data.shape, type(output_data))
#output_data = (output_data-133)*0.04431682452559471
print(output_data.shape)
y = output_data.flatten()
y = y/np.linalg.norm(y)



interpreter2 = tf.lite.Interpreter(model_path="/home/fangyu/fy/tflite/fitune_insightface/output/test.tflite")
interpreter2.allocate_tensors()

# Get input and output tensors.
input_details2 = interpreter2.get_input_details()
output_details2 = interpreter2.get_output_details()

print(input_details2)
print(output_details2)

#image2 = np.ones((1,112,112,3))
image_2 = image.astype('float32')
interpreter2.set_tensor(input_details2[0]['index'], image_2)
interpreter2.invoke()
output_data2 = interpreter2.get_tensor(output_details2[0]['index'])
print(output_data2.shape, type(output_data2))
#output_data2 = (output_data2-121)*0.03864818066358566
print(output_data2.shape)
y2 = output_data2.flatten()
y2 = y2/np.linalg.norm(y2)


dis = y.dot(y2.T)
print(dis)