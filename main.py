import matplotlib.image as mpimg
from tensorflow.keras.utils import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

model = load_model('model12.h5')

image_path = 'chest_xray\\test\\PNEUMONIA\\test_pneu_1305.jpg'
image = mpimg.imread(image_path)

test_image = load_img(image_path, target_size = (224, 224))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
predict = model.predict(test_image)
predict = np.argmax(predict, axis=-1)

if predict == 0:
    prediction = 'Normal'
else:
    prediction = 'Pneumonia +VE'

plt.imshow(image);plt.suptitle(prediction, fontsize = 20);plt.axis("off");plt.show()
