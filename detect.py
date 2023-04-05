import matplotlib.image as mpimg
from tensorflow.keras.utils import img_to_array, load_img
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# Load the model
model = load_model('model12.h5')

# Convert the model to a quantized model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
quantized_model = converter.convert()

# Save the quantized model
with open("quantized_model.tflite", "wb") as f:
    f.write(quantized_model)

# Load the quantized model
interpreter = tf.lite.Interpreter(model_content=quantized_model)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image_path = 'chest_xray\\test\\PNEUMONIA\\test_pneu_130.jpg'
image = mpimg.imread(image_path)

test_image = load_img(image_path, target_size = (224, 224))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
pred = model.predict(test_image)
predict = np.argmax(pred, axis=-1)

if predict == 0:
    prediction = 'Normal'
else:
    prediction = 'Pneumonia +VE'

plt.imshow(image);plt.suptitle(prediction, fontsize = 20);plt.axis("off");plt.show()