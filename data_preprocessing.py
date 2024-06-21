import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the root directory of your dataset
dataset_dir = "C:\\Users\\856ma\\Downloads\\archive\\images"

# Define the directories for training, validation, and test sets
train_dir = os.path.join(dataset_dir, 'train')
validation_dir = os.path.join(dataset_dir, 'validation')
test_dir = os.path.join(dataset_dir, 'test')

# Parameters for ImageDataGenerator
batch_size = 32
target_size = (150, 150)

# Use ImageDataGenerator for data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary'  # Adjust class_mode as per your dataset
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary'  # Adjust class_mode as per your dataset
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary'  # Adjust class_mode as per your dataset
)

# Print class indices
print("Class indices:")
print(train_generator.class_indices)

# Optionally, you can further process your data or build a model here

print("Data preprocessing completed successfully.")
