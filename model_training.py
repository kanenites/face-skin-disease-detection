import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define paths
base_dir = os.path.join(os.path.dirname(__file__), 'DATA')
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'testing')

# Check if train and test directories exist
if not os.path.exists(train_dir):
    print(f"Error: The directory {train_dir} does not exist. Please verify the path.")
if not os.path.exists(test_dir):
    print(f"Error: The directory {test_dir} does not exist. Please verify the path.")

# Define image dimensions
img_height, img_width = 150, 150

# Initialize ImageDataGenerator for training (with augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,  # Increased rotation
    width_shift_range=0.3,  # Increased width shift
    height_shift_range=0.3,  # Increased height shift
    shear_range=0.2,  # Added shear
    zoom_range=0.2,  # Increased zoom range
    horizontal_flip=True,
    fill_mode='nearest'
)

# Initialize ImageDataGenerator for testing (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical'
)

# Load testing images (if the directory exists)
if os.path.exists(test_dir):
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical'
    )

# Define model using transfer learning with VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Unfreeze the last few layers for fine-tuning
for layer in base_model.layers[-4:]:  # Unfreeze the last few layers
    layer.trainable = True

# Add custom layers
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)  # Add Dropout layer
output = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile the model with a lower learning rate
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

# Set up callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Train the model
history = model.fit(train_generator, 
                    epochs=50, 
                    validation_data=test_generator, 
                    callbacks=[early_stopping, reduce_lr])

# Save the model to the same directory (base_dir)
model_save_path = os.path.join(base_dir, 'disease_classification_model.h5')
model.save(model_save_path)

print(f"Model saved to {model_save_path}")
