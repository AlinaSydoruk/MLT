import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
import matplotlib.pyplot as plt
import os

url = 'https://drive.google.com/drive/folders/1nzVk4GOvKR6P87uPszUkKMPtaXV_wrZf'

# Save datasets path
PATH = "./ML/unzipped_folder/"
PATH_TRAIN = f"{PATH}train_images/"
PATH_TEST = f"{PATH}test_images/"

# Check datasets in file system

os.listdir(PATH)

# Set the correct local path where you want to store your images
train_images_path = os.path.join(PATH, "train_images")
test_images_path = os.path.join(PATH, "test_images")

# Create subfolders for train and test datasets if they don't exist
if not os.path.exists(train_images_path):
    os.mkdir(train_images_path)
if not os.path.exists(test_images_path):
    os.mkdir(test_images_path)
print("Folders created or already exist.")

# Corrected path for unzipping the train dataset
train_images_path = os.path.join(PATH, "train_images")
import zipfile
with zipfile.ZipFile(train_images_path, "r") as archive:
    for file in archive.namelist():
        archive.extract(file, train_images_path)

# Define paths
test_images_path = os.path.join(PATH, "test_images")
# Unzip test dataset
with zipfile.ZipFile(test_images_path, "r") as archive:
    for file in archive.namelist():
        archive.extract(file, test_images_path)

# Save our filenames

animals = pd.DataFrame({
    'Image name': imagenames,
    'Category': categories
})
animals.head(5)

# Check total amount of 0 and 1 labels
animals['Category'].value_counts()

# Draw a cat

# Don't forget to install 'Pillow' module (conda install pillow) to give a 'pyplot' ability of working with '.jpg'
img = plt.imread(f"{PATH_TRAIN}train/{imagenames[1]}")
plt.imshow(img)

# Split data on train and validation subsets
from sklearn.model_selection import train_test_split
X_train, X_val = train_test_split(animals, test_size=0.2, random_state=2)
X_train = X_train.reset_index()
X_val = X_val.reset_index()

# We may want to use only 1800 images because of CPU computational reasons. If so, this code should be run
X_train = X_train.sample(n=1800).reset_index()
X_val = X_val.sample(n=180).reset_index()

# Count
total_X_train = X_train.shape[0]
total_X_val = X_val.shape[0]
total_X_train, total_X_val

# By default, the VGG16 model expects images with input the size 224 x 224 pixels with 3 channels (e.g., color).
image_size = 224
input_shape = (image_size, image_size, 3)

# Define CNN model constants
epochs = 5
batch_size = 16

# Нормалізація даних
train_images = train_images / 255.0
test_images = test_images / 255.0

# Перетворення зображень у одновимірні вектори
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

# Створення моделі
model_fcn = Sequential([
    Dense(512, activation='relu', input_shape=(train_images.shape[1],)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Компіляція моделі
model_fcn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Навчання моделі
model_fcn.fit(train_images, train_labels, epochs=10, validation_split=0.2)

# Оцінка моделі
test_loss, test_acc = model_fcn.evaluate(test_images, test_labels)
print(f"Точність на тестових даних: {test_acc:.2f}")



# Створення моделі
model_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Компіляція моделі
model_cnn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Навчання моделі
model_cnn.fit(train_images, train_labels, epochs=10, validation_split=0.2)





# Оцінка моделі
test_loss, test_acc = model_cnn.evaluate(test_images, test_labels)
print(f"Точність на тестових даних: {test_acc:.2f}")



# Завантаження моделі VGG19
base_model_vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model_vgg19.trainable = False  # Заморожування моделі

# Додавання нових шарів
model_vgg19 = Sequential([
    base_model_vgg19,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Компіляція моделі
model_vgg19.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Навчання моделі
history_vgg19 = model_vgg19.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Завантаження моделі ResNet50
base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model_resnet.trainable = False  # Заморожування моделі

# Додавання нових шарів
model_resnet = Sequential([
    base_model_resnet,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Компіляція моделі
model_resnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Навчання моделі
history_resnet = model_resnet.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

def plot_history(histories):
    plt.figure(figsize=(14, 5))
    for history, model_name in histories:
        plt.plot(history.history['accuracy'], label=f'{model_name} Train Acc')
        plt.plot(history.history['val_accuracy'], label=f'{model_name} Val Acc')

    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    return plt

# Повторне навчання FCN та CNN з більшою кількістю епох
history_fcn_more = model_fcn.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))
history_cnn_more = model_cnn.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))

# Візуалізація
plot_history([(history_fcn_more, 'FCN'), (history_cnn_more, 'CNN')]).show()