import numpy as np
import os
from random import shuffle
import glob
import cv2
from model import get_model
import tflearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score
import matplotlib.pyplot as plt

from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator

# Constants
path = Path('./ECUSTFD-resized--master')
IMG_SIZE = 400
LR = 1e-3
MODEL_NAME = f'Food_Calorie_detector-{LR}-5conv-basic'
no_of_fruits = 7
percentage = 0.3
no_of_images = 100
batch_size = 32

# Create and save training data
def create_train_data(path):
    training_data = []
    folders = os.listdir(path)[:no_of_fruits]
    for i, folder in enumerate(folders):
        label = [0] * no_of_fruits
        label[i] = 1
        print(folder)
        k = 0
        for j in glob.glob(os.path.join(path, folder, '*.jpg')):
            if k == no_of_images:
                break
            k += 1
            img = cv2.imread(j)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.savez(f'training_{no_of_fruits}_{no_of_images}_{IMG_SIZE}.npz', 
             images=np.array([i[0] for i in training_data]), 
             labels=np.array([i[1] for i in training_data]))
    return training_data

# Create training data
training_data = create_train_data(path)

# Load training data
try:
    data = np.load(f'training_{no_of_fruits}_{no_of_images}_{IMG_SIZE}.npz')
    X = data['images']
    Y = data['labels']
except FileNotFoundError:
    print("Training data file not found!")
    exit()

# Split data into training and testing sets
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define a separate validation dataset
validation_split = 0.2  # 20% of the data for validation
validation_indices = np.random.choice(len(train_X), size=int(validation_split * len(train_X)), replace=False)
validation_X = train_X[validation_indices]
validation_Y = train_Y[validation_indices]

# Remove validation data from training data
train_X = np.delete(train_X, validation_indices, axis=0)
train_Y = np.delete(train_Y, validation_indices, axis=0)

# Data Augmentation for Training
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define a function to generate augmented data
def generate_augmented_data(X, Y, batch_size):
    num_samples = X.shape[0]
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_X = X[offset:offset+batch_size]
            batch_Y = Y[offset:offset+batch_size]
            # Generate augmented images and labels
            for i in range(len(batch_X)):
                x = batch_X[i]
                y = batch_Y[i]
                x_aug, y_aug = train_datagen.flow(np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)).next()
                batch_X[i] = x_aug.squeeze()
                batch_Y[i] = y_aug.squeeze()
            yield batch_X, batch_Y

# Train the model
model = get_model(IMG_SIZE, no_of_fruits, LR) 

class TrainingHistory(tflearn.callbacks.Callback):
    def __init__(self, model, test_X, test_Y):
        super(TrainingHistory, self).__init__()
        self.model = model
        self.test_X = test_X
        self.test_Y = test_Y
        self.history = {'acc': [], 'val_acc': [], 'test_acc': []}  # Initialize history

    def on_epoch_end(self, training_state):
        logs = training_state.val_loss if hasattr(training_state, 'val_loss') else training_state.loss
        self.history['acc'].append(training_state.acc_value)

        # Calculate validation accuracy manually
        val_pred = self.model.predict(validation_X)
        val_pred_classes = np.argmax(val_pred, axis=1)
        val_true_classes = np.argmax(validation_Y, axis=1)
        val_acc = np.mean(val_pred_classes == val_true_classes)
        self.history['val_acc'].append(val_acc)

        # Calculate test accuracy manually
        test_pred = self.model.predict(self.test_X)
        test_pred_classes = np.argmax(test_pred, axis=1)
        test_true_classes = np.argmax(self.test_Y, axis=1)
        test_acc = np.mean(test_pred_classes == test_true_classes)
        self.history['test_acc'].append(test_acc)

# Create instances of the callbacks
training_history = TrainingHistory(model, test_X, test_Y)

# Train the model with the custom callback
model.fit({'input': train_X}, {'targets': train_Y}, n_epoch=10, 
          validation_set=({'input': validation_X}, {'targets': validation_Y}),
          batch_size=batch_size, show_metric=True, validation_batch_size=batch_size,
          run_id=MODEL_NAME, callbacks=[training_history])

# Save the model
model_save_at = Path("model") / MODEL_NAME
model.save(str(model_save_at))
print("Food_Calorie_detector model saved:", model_save_at)

# Plot training, validation, and test accuracies
plt.figure(figsize=(10, 5))
plt.plot(training_history.history['acc'], label='Training Accuracy')
plt.plot(training_history.history['val_acc'], label='Validation Accuracy')
plt.plot(training_history.history['test_acc'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training, Validation, and Test Accuracy')
plt.legend()
plt.show()
