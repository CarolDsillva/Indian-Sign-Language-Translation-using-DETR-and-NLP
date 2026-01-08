"""
Sign Language Model Architecture and Loader

This file contains the model architecture definition.
Use this to load your model weights without compatibility issues.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def create_sign_language_model():
    """
    Creates the Indian Sign Language model architecture.
    
    Returns:
        tf.keras.Model: Compiled model ready for weight loading
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(64, 64, 3)),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='valid', name='conv2d'),
        layers.MaxPooling2D((2, 2), name='max_pooling2d'),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv2d_1'),
        layers.MaxPooling2D((2, 2), name='max_pooling2d_1'),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='valid', name='conv2d_2'),
        layers.MaxPooling2D((2, 2), name='max_pooling2d_2'),
        
        # Dense layers
        layers.Flatten(name='flatten'),
        layers.Dense(512, activation='relu', name='dense'),
        layers.Dropout(0.5, name='dropout'),
        layers.Dense(35, activation='softmax', name='dense_1')  # 35 classes: 0-9 + A-Z
    ], name='sign_language_model')
    
    return model


def load_model(weights_path='model.weights.h5'):
    """
    Load the sign language model with weights.
    
    Args:
        weights_path (str): Path to the weights file
        
    Returns:
        tf.keras.Model: Model with loaded weights
    """
    # Create model
    model = create_sign_language_model()
    
    # Build it
    model.build((None, 64, 64, 3))
    
    # Load weights
    model.load_weights(weights_path)
    
    print(f"✅ Model loaded successfully from {weights_path}")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Total parameters: {model.count_params():,}")
    
    return model


# Label mapping for predictions
LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Digits
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',  # Letters A-J
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',  # Letters K-T
    'U', 'V', 'W', 'X', 'Y', 'Z'                        # Letters U-Z
]


def predict_sign(model, image):
    """
    Predict sign language character from image.
    
    Args:
        model: Loaded Keras model
        image: Preprocessed image (64x64x3, normalized)
        
    Returns:
        tuple: (predicted_label, confidence)
    """
    import numpy as np
    
    # Ensure correct shape
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # Predict
    prediction = model.predict(image, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    
    return LABELS[class_idx], confidence


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    print("="*60)
    print("Sign Language Model Loader")
    print("="*60)
    
    # Load model
    model = load_model('model.weights.h5')
    
    # Test with random input
    print("\n🧪 Testing with random input...")
    test_input = np.random.rand(1, 64, 64, 3).astype('float32')
    label, confidence = predict_sign(model, test_input)
    
    print(f"   Predicted: {label}")
    print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    print("\n✅ Model is ready to use!")
    print("="*60)
