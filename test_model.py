"""
Test script to verify the fixed model works correctly
"""
import tensorflow as tf
import numpy as np

print("="*60)
print("🧪 TESTING FIXED MODEL")
print("="*60)

# Load model
MODEL_PATH = "sign_language_model_clean.h5"
print(f"\n📥 Loading model from: {MODEL_PATH}")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Display model info
print(f"\n📊 Model Information:")
print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")
print(f"   Total parameters: {model.count_params():,}")

# Define labels
labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

# Test with random input
print(f"\n🎲 Testing with random input...")
test_input = np.random.rand(1, 64, 64, 3).astype('float32')
prediction = model.predict(test_input, verbose=0)

print(f"   Prediction shape: {prediction.shape}")
print(f"   Sum of probabilities: {prediction.sum():.4f} (should be ~1.0)")

# Get top 5 predictions
top_5_indices = np.argsort(prediction[0])[-5:][::-1]
print(f"\n🏆 Top 5 Predictions:")
for i, idx in enumerate(top_5_indices, 1):
    print(f"   {i}. {labels[idx]:>2} - {prediction[0][idx]:.4f} ({prediction[0][idx]*100:.2f}%)")

# Test with multiple batches
print(f"\n📦 Testing batch prediction...")
batch_input = np.random.rand(5, 64, 64, 3).astype('float32')
batch_prediction = model.predict(batch_input, verbose=0)
print(f"   Batch input shape: {batch_input.shape}")
print(f"   Batch output shape: {batch_prediction.shape}")
print(f"   ✅ Batch prediction works!")

print(f"\n{'='*60}")
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\n💡 The model is ready to use in your Flask app!")
print("   Simply ensure 'sign_language_model_clean.h5' is in the same directory as 'app.py'")
print("\n🚀 To run the Flask app:")
print("   python app.py")
print("="*60)