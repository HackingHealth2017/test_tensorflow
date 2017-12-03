from keras.applications import InceptionV3
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
from keras.layers import Input

model = InceptionV3(weights='imagenet')

img = image.load_img('test.jpg', target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
for i in range(5):
    print('Predicted:', decode_predictions(preds, top=2)[0][i])