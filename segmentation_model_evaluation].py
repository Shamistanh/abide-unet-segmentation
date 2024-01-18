import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('unet_segmentation.h5')  # Replace with the actual path to your trained model

def preprocess_image(image_path, target_size=(512, 512)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Path to the new image for inference
new_image_path = '/Users/shamistanhuseynov/PycharmProjects/abide_segmentation/dataset/main/1.png'
# Preprocess the new image
input_image = preprocess_image(new_image_path)

# Perform inference
prediction = model.predict(input_image)

# Threshold the predicted mask (assuming binary segmentation)
threshold = 0.5
predicted_mask = (prediction > threshold).astype(np.uint8)

# Display the original image, ground truth, and predicted mask
original_image = img_to_array(load_img(new_image_path))
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(original_image.astype(np.uint8))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(original_image.astype(np.uint8))  # Placeholder for ground truth
plt.title('Ground Truth')
plt.axis('off')

# Predicted Mask
plt.subplot(1, 3, 3)
plt.imshow(predicted_mask[0, :, :, 0], cmap='gray')
plt.title('Predicted Mask')
plt.axis('off')

plt.show()
