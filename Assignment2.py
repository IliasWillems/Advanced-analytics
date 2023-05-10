import shutil
import random
import numpy as np
import json
import zipfile
import os
import pandas as pd
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import os
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Dense, Flatten
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import lime
from lime import lime_image
from PIL import Image


# paths to zip file of images and json dataset
# zip_path = "path/to/images.zip"
# dataset_path = "path/to/dataset.json"

# IK HEB DIE OP EEN ANDERE PLEK STEKEN DAAROM IS HET DIT
zip_path = r"D:/school/Advanced Analytics is a Big Data World/images.zip"
dataset_path = r"D:/school/Advanced Analytics is a Big Data World/dataset.json"

# From google drive
#zip_path = '/content/drive/MyDrive/path/to/images.zip'
#dataset_path = '/content/drive/MyDrive/path/to/dataset.json'

# extract images from zip file
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    all_filenames = zip_ref.namelist()
    random.shuffle(all_filenames)
    # number of images used can be changed here
    selected_filenames = all_filenames[:1000]
    zip_ref.extractall("path/to/extracted_images", members=selected_filenames)
images_path = "path/to/extracted_images"
# load the JSON file and extract the price_category for each image
with open(dataset_path, "r") as f:
    data = json.load(f)
price_categories = {}
image_ids = []
for item in data:
    # Check if 'price_category' key is present in the 'item' dictionary
    if 'price_category' in item and item['price_category'] is not None:
        # Extract the price category
        price_category = item['price_category']['label']
        # Extract the image IDs from the "full_images" list and map them to their price category
        for img in item["more_details"]["full_images"]:
            image_id = img["image_id"]
            if f"{image_id}.jpg" in selected_filenames:
                image_ids.append(image_id)
                price_categories[image_id] = price_category

# now I have a list of image id's and a library of price categories
# the file names of the images are just the id with .jpg
# I want to create the training, validation and test set.
# For this I need to make a pandas data frame with two columns, with the image file names and one with the categories
random.shuffle(image_ids)  # shuffle image id's

train_size = 0.7
val_size = 0.2
test_size = 0.1

train_image_ids = image_ids[:int(len(image_ids) * train_size)]
val_image_ids = image_ids[int(len(image_ids) * train_size):int(len(image_ids) * (train_size + val_size))]
test_image_ids = image_ids[int(len(image_ids) * (train_size + val_size)):]

# Get lists of image filenames for each set
train_filenames = [f"{image_id}.jpg" for image_id in train_image_ids]
val_filenames = [f"{image_id}.jpg" for image_id in val_image_ids]
test_filenames = [f"{image_id}.jpg" for image_id in test_image_ids]

# Create a pandas dataframe with two columns: "filename" and "price_category"
train_df = pd.DataFrame(
    {"filename": train_filenames, "price_category": [price_categories[image_id] for image_id in train_image_ids]})
val_df = pd.DataFrame(
    {"filename": val_filenames, "price_category": [price_categories[image_id] for image_id in val_image_ids]})
test_df = pd.DataFrame(
    {"filename": test_filenames, "price_category": [price_categories[image_id] for image_id in test_image_ids]})

# Set the paths for the training, validation, and test data directories
train_data_dir = os.path.join(images_path, "train")
val_data_dir = os.path.join(images_path, "val")
test_data_dir = os.path.join(images_path, "test")

# Create the directories if they don't exist already
os.makedirs(train_data_dir, exist_ok=True)
os.makedirs(val_data_dir, exist_ok=True)
os.makedirs(test_data_dir, exist_ok=True)

# Copy the image files to the appropriate directories
for filename in train_filenames:
    src_path = os.path.join(images_path, filename)
    dst_path = os.path.join(train_data_dir, filename)
    shutil.copy(src_path, dst_path)

for filename in val_filenames:
    src_path = os.path.join(images_path, filename)
    dst_path = os.path.join(val_data_dir, filename)
    shutil.copy(src_path, dst_path)

for filename in test_filenames:
    src_path = os.path.join(images_path, filename)
    dst_path = os.path.join(test_data_dir, filename)
    shutil.copy(src_path, dst_path)

# Define the image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Create an ImageDataGenerator for data preprocessing including rotation augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=20, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Generate the training, validation, and testing datasets
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='categorical')
val_generator = val_datagen.flow_from_directory(val_data_dir, target_size=(img_width, img_height),
                                                batch_size=batch_size, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(img_width, img_height),
                                                  batch_size=batch_size, class_mode='categorical')

# Define the CNN architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit_generator(train_generator, steps_per_epoch=train_generator.samples // batch_size, epochs=50,
                              validation_data=val_generator, validation_steps=val_generator.samples // batch_size)

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.samples // batch_size)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# interpretability technique LIME to get more insights on the decisions
explainer = lime_image.LimeImageExplainer()


def predict_wrapper(images):
    # This function is a wrapper around the model's prediction function
    # It takes in a batch of images (N, 224, 224, 3) and returns the predicted probabilities (N, 5)
    return model.predict(images)


# Get an example image from the validation set
example_image_path = os.path.join(val_data_dir, val_filenames[0])
example_image = Image.open(example_image_path)

# Explain the model's prediction for the example image
explanation = explainer.explain_instance(np.array(example_image), predict_wrapper, top_labels=5)

# Show the LIME visualization for the explanation
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
img_boundry = mark_boundaries(temp/2 + 0.5, mask)
Image.fromarray((img_boundry*255).astype(np.uint8))

# pretrained model VGG16

# preprocessing according to VGG16
batch_size = 64
train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=90,
                                     brightness_range=[0.1, 0.7],
                                     width_shift_range=0.5,
                                     height_shift_range=0.5,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     validation_split=0.15,
                                     preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(rescale=1. / 255,preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(rescale=1. / 255,preprocessing_function=preprocess_input)

# Generate the training, validation, and testing datasets (I followed the instructions on https://www.learndatasci.com/tutorials/hands-on-transfer-learning-keras/)
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='categorical', shuffle=True,
                                               seed=42)
val_generator = val_datagen.flow_from_directory(val_data_dir, target_size=(img_width, img_height),
                                                batch_size=batch_size, class_mode='categorical', shuffle=True,
                                               seed=42)
test_generator = test_datagen.flow_from_directory(test_data_dir,
                                             target_size=(img_width, img_height),
                                             class_mode='categorical',
                                             batch_size=1,
                                             shuffle=False,
                                             seed=42)

# Load without the top layer input_shape(img_width, img_height, 3) (the 3 is for the colors rbg)
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# make sure we don't train these layers
for layer in vgg_model.layers:
    layer.trainable = False

# Add a new output layer with 4 classes
x = Flatten()(vgg_model.output)
output_layer = Dense(4, activation='softmax')(x)

# create a new model with the pre-trained layers and the new output layer and train
model_VGG = Model(inputs=vgg_model.input, outputs=output_layer)

model_VGG.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model_VGG.fit_generator(train_generator, steps_per_epoch=train_generator.samples // batch_size, epochs=5,
                              validation_data=val_generator, validation_steps=val_generator.samples // batch_size)

# evaluate
test_loss, test_acc = model_VGG.evaluate_generator(test_generator, steps=test_generator.samples // batch_size)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

#LIME
explainer = lime_image.LimeImageExplainer()


def predict_wrapper_VGG(images):
    # This function is a wrapper around the model's prediction function
    # It takes in a batch of images (N, 224, 224, 3) and returns the predicted probabilities (N, 5)
    return model_VGG.predict(images)


# Get an example image from the validation set
example_image_path = os.path.join(val_data_dir, val_filenames[0])
example_image = Image.open(example_image_path)

# Explain the model's prediction for the example image
explanation = explainer.explain_instance(np.array(example_image), predict_wrapper_VGG, top_labels=5)

# Show the LIME visualization for the explanation
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
img_boundry = mark_boundaries(temp/2 + 0.5, mask)
Image.fromarray((img_boundry*255).astype(np.uint8))
