import os
import shutil
import random

# Define the root directory of your dataset
dataset_dir = "C:\\Users\\856ma\\Downloads\\archive\\images"

# Define the directories for training, validation, and test sets
train_dir = os.path.join(dataset_dir, 'train')
validation_dir = os.path.join(dataset_dir, 'validation')
test_dir = os.path.join(dataset_dir, 'test')

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Shuffle the list of image files
image_files = os.listdir(dataset_dir)
random.shuffle(image_files)

# Define the split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Calculate the number of images for each set
num_images = len(image_files)
num_train = int(num_images * train_ratio)
num_val = int(num_images * val_ratio)
num_test = num_images - num_train - num_val

# Divide the dataset into training, validation, and test sets
train_images = image_files[:num_train]
val_images = image_files[num_train:num_train + num_val]
test_images = image_files[num_train + num_val:]

# Move images to respective directories
def move_images(image_list, dest_dir):
    for image in image_list:
        src_path = os.path.join(dataset_dir, image)
        dest_path = os.path.join(dest_dir, image)
        if os.path.isfile(src_path):  # Check if the source is a file
            shutil.move(src_path, dest_path)

move_images(train_images, train_dir)
move_images(val_images, validation_dir)
move_images(test_images, test_dir)
