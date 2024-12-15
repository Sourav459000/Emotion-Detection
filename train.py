import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Set dataset paths
train_dir = 'data/train'
test_dir = 'data/test'

# Hyperparameters
img_width, img_height = 48, 48
batch_size = 16
epochs = 50
initial_learning_rate = 0.001
fine_tune_learning_rate = 0.00005

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Reduced rotation range
    width_shift_range=0.1,  # Reduced shift range
    height_shift_range=0.1,
    brightness_range=[0.9, 1.1],  # Subtle brightness adjustment
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Compute class weights
class_weights = None  # Disable class weights for this iteration

# Load pre-trained MobileNetV2 model + higher layers
base_model = MobileNetV2(input_shape=(img_width, img_height, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model initially

# Add custom layers
inputs = Input(shape=(img_width, img_height, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # Reduced L2 regularization
x = BatchNormalization()(x)
x = Dropout(0.3)(x)  # Reduced dropout rate
outputs = Dense(7, activation='softmax')(x)
model = Model(inputs, outputs)

# Compile the model
model.compile(optimizer=Adam(learning_rate=initial_learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the model (Initial Phase)
history = model.fit(
    train_generator,
    epochs=epochs // 2,
    validation_data=test_generator,
    callbacks=[lr_scheduler]
)

# Fine-tuning
base_model.trainable = True  # Unfreeze the base model
for layer in base_model.layers[:50]:  # Keep first 50 layers frozen
    layer.trainable = False

# Recompile with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=fine_tune_learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model (Fine-tuning Phase)
fine_tune_history = model.fit(
    train_generator,
    epochs=epochs,
    initial_epoch=epochs // 2,
    validation_data=test_generator,
    callbacks=[lr_scheduler]
)

# Save the model for deployment
model.save('emotion_recognition_model.h5', save_format='h5')

print("Model training complete. Saved as 'emotion_recognition_model.h5'")