import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from calories import calories
from model import get_model


IMG_SIZE = 400
LR = 1e-3
no_of_fruits = 7

MODEL_NAME = f'Food_Calorie_detector-{LR}-5conv-basic.model'
model_save_at = os.path.join("model", MODEL_NAME)

model = get_model(IMG_SIZE, no_of_fruits, LR)
model.load(model_save_at)

labels = list(np.load('labels_list.npy'))

test_data = 'test_images/mango.JPG'
img = cv2.imread(test_data)
img1 = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

model_out = model.predict([img1])
result = np.argmax(model_out)
name = labels[result]
cal = round(calories(result + 1, img), 2)

plt.imshow(img)
plt.title(f'{name}({cal}kcal)')
plt.axis('off')
plt.show()
