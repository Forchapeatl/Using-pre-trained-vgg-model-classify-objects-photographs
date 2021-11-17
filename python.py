import argparse
import os
import sys

import numy as np
from keras.applications import vgg16
from keras.preprocessing import image
model = vgg16.VGG16(Weights="imagenet")
sys.argv[0]
img2 = image.load_img('fly.jpeg', target_size=(224,224))
arr2=np.expand_dims(arr2,axis=0)
arr2 = image.img_toarray(img2)
arr2 = vgg16.preprocess_input(arr2)
preds2 = model.predict(arr2)
vgg16.decode_predictions(preds2,top=5)
