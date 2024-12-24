import pandas as pd
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.applications import VGG19, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D


#url = 'https://drive.google.com/drive/folders/1nzVk4GOvKR6P87uPszUkKMPtaXV_wrZf'

# Save datasets path

# Define the paths to the training and validation directories
base_dir = 'C:/Users/Alina Sidoruk/PycharmProjects/ML_Lab1/data'
validation_dir = f'{base_dir}/test'
train_dir = f'{base_dir}/train'
validation_dir = f'{base_dir}/val'

import tensorflow as tf
print(tf.__version__)


# ImageDataGenerator for normalization and augmentation
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load images and apply transformations
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=20,
    class_mode='binary')

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),
    batch_size=20,
    class_mode='binary')

# Model A: Fully Connected Network
model_dense = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Model B: Convolutional Neural Network
model_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train Dense Model
model_dense.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_dense.fit(train_generator, validation_data=validation_generator, epochs=10)

# Compile and train CNN Model
model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_cnn.fit(train_generator, validation_data=validation_generator, epochs=10)


# Завантаження та налаштування моделей VGG19 та ResNet для перенесення навчання
base_vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Заморожування конволюційних шарів
for layer in base_vgg19.layers:
    layer.trainable = False
for layer in base_resnet.layers:
    layer.trainable = False

# Додавання нових шарів
x_vgg = GlobalAveragePooling2D()(base_vgg19.output)
x_res = GlobalAveragePooling2D()(base_resnet.output)
x_vgg = Dense(512, activation='relu')(x_vgg)
x_res = Dense(512, activation='relu')(x_res)
output_vgg = Dense(1, activation='sigmoid')(x_vgg)
output_res = Dense(1, activation='sigmoid')(x_res)

model_vgg19 = Model(inputs=base_vgg19.input, outputs=output_vgg)
model_resnet = Model(inputs=base_resnet.input, outputs=output_res)

# Компіляція і тренування адаптованих моделей
model_vgg19.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_resnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_vgg19.fit(train_generator, validation_data=validation_generator, epochs=10)
model_resnet.fit(train_generator, validation_data=validation_generator, epochs=10)

# Оцінка продуктивності моделей
dense_scores = model_dense.evaluate(validation_generator)
cnn_scores = model_cnn.evaluate(validation_generator)
vgg_scores = model_vgg19.evaluate(validation_generator)
res_scores = model_resnet.evaluate(validation_generator)

print("Performance of Dense Model:", dense_scores)
print("Performance of CNN Model:", cnn_scores)
print("Performance of VGG19:", vgg_scores)
print("Performance of ResNet:", res_scores)