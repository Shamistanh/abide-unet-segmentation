import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Define paths
images_path = '/Users/shamistanhuseynov/PycharmProjects/abide_segmentation/dataset/main'
annotations_path = '/Users/shamistanhuseynov/PycharmProjects/abide_segmentation/dataset/annotations'

# Image dimensions
input_height, input_width = 512, 512
num_classes = 2  # Background and brain

# Load images and annotations
image_files = os.listdir(images_path)
annotation_files = os.listdir(annotations_path)

# Sort files to ensure images and annotations are matched correctly
image_files.sort()
annotation_files.sort()

# Create empty arrays to store images and annotations
images = np.zeros((len(image_files), input_height, input_width, 3), dtype=np.float32)
annotations = np.zeros((len(annotation_files), input_height, input_width, 1), dtype=np.float32)

# Load images and annotations into arrays
for i, (image_file, annotation_file) in enumerate(zip(image_files, annotation_files)):
    image_path = os.path.join(images_path, image_file)
    annotation_path = os.path.join(annotations_path, annotation_file)

    img = load_img(image_path, target_size=(input_height, input_width))
    images[i] = img_to_array(img) / 255.0

    annotation = load_img(annotation_path, target_size=(input_height, input_width), color_mode='grayscale')
    annotations[i] = img_to_array(annotation) / 255.0

# Split the dataset into training and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(images, annotations, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

# Build the U-Net model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(input_height, input_width, 3), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (1, 1), activation='sigmoid'))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Define a checkpoint to save the best model
checkpoint = ModelCheckpoint('unet_segmentation.h5', save_best_only=True)

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_val, y_val), callbacks=[checkpoint])

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}')

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Assuming you already have your model trained and stored in the 'model' variable

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions (0 or 1)

# Generate a confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))
