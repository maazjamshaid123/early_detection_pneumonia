import streamlit as st
import matplotlib.image as mpimg
from tensorflow.keras.utils import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

@st.cache(allow_output_mutation=True)
def load_keras_model(model_file):
    return load_model(model_file)

model_file = 'model12.h5'
model = load_keras_model(model_file)

st.title('Image Classification App')

image_path = st.text_input('Enter the path to the image file:')

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

st.image(image, caption=prediction, use_column_width=True)
