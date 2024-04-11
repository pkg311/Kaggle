import numpy as np
import cv2
import tensorflow as tf

# Load the trained model
final_model = tf.keras.models.load_model('my_model.weights.h5')
class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']  # Update with your class names

def preprocess_image(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (260, 260))
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize pixel values
    return image

def predict_leaf_health(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    
    # Make predictions
    predictions = final_model.predict(preprocessed_image)
    
    # Get the predicted class
    predicted_class = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class]
    
    # Get the probability of the predicted class
    predicted_probability = predictions[0][predicted_class]
    
    return predicted_class_name, predicted_probability

def main():
    # Path to the input image
    image_path = 'corn_leaf.jpg'  # Update with your image path
    
    # Predict leaf health
    predicted_class, predicted_probability = predict_leaf_health(image_path)
    
    # Print the prediction
    print("Predicted Class:", predicted_class)
    print("Predicted Probability:", predicted_probability)

if __name__ == "__main__":
    main()
