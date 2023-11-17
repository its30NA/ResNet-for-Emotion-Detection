import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

import tensorflow as tf
tf.config.run_functions_eagerly(True)
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.size'] = 10

train_path = Path("E:/project/start_rasberrypi/archive/train")
test_path = Path("E:/project/start_rasberrypi/archive/test")
random_seed = 42

img_size = 48
batch_size = 64
train = tf.keras.utils.image_dataset_from_directory(
  train_path,
  # validation_split=0.2,
  # subset="training",
  # seed=123,
  image_size=(img_size, img_size),
  batch_size=batch_size,
  label_mode ="categorical"
)

val = tf.keras.utils.image_dataset_from_directory(
  test_path,
  # validation_split=0.2,
  # subset="validation",
  # seed=123,
  image_size=(img_size, img_size),
  batch_size=batch_size,
  label_mode ="categorical"
)

class_names = train.class_names

names = []
for name in class_names:
  # names.append(name.split("-")[1])
  names.append(name)

print(names[:10])  # Printing some species

plt.figure(figsize=(10, 10))
for images, labels in train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")

Model_URL ='https://kaggle.com/models/google/resnet-v2/frameworks/TensorFlow2/variations/50-classification/versions/2'
model = Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_size, img_size, 3)),
    hub.KerasLayer(Model_URL),
    tf.keras.layers.Dense(7, activation="softmax")])

model.compile(
      loss=tf.keras.losses.CategoricalCrossentropy(),
      optimizer=tf.keras.optimizers.Adam(),
      metrics=["accuracy"]
  )

model.build((img_size, img_size, 3))

model.summary()

model_name = "model.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(model_name,
                                                monitor="val_loss",
                                                mode="min",
                                                save_best_only=True,
                                                verbose=1)

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                                                 verbose=1, restore_best_weights=True)

reduce_lr = tf._keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                  patience=5, min_lr=0.0001)

history = model.fit(train, epochs=30, validation_data=val, callbacks=[checkpoint,earlystopping, reduce_lr])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()

loss, accuracy = model.evaluate(val)

print(f"Accuracy is: {round(accuracy*100,2)}%")