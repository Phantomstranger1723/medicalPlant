import os
import numpy as np
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

main_data_dir = 'dataset/Segmented Medicinal Leaf Images'
batch_size = 32
epochs = 10

# List all subdirectories (class folders) in the main directory
class_folders = os.listdir(main_data_dir)
num_classes = len(class_folders)

# Use LabelEncoder to encode class labels
label_encoder = LabelEncoder()
label_encoder.fit(class_folders)

# Load and preprocess images
X = []
y = []
for class_folder in class_folders:
    class_path = os.path.join(main_data_dir, class_folder)
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        X.append(img_array)
        y.append(class_folder)

X = np.array(X)
y = label_encoder.transform(y)  # Encode string labels to integers
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Load MobileNetV2 base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)  # Adding dropout for regularization
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val)
)

model.save('plant_identification_model2.h5')