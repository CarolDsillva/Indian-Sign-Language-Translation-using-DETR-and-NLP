from flask import Flask, render_template, jsonify, Response, request
import cv2
import numpy as np
import tensorflow as tf
import os
import string
import speech_recognition as sr
from flask import request
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# Import your DETR utilities
from model import DETR
from utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors

app = Flask(__name__)

# # ============================
# # ✅ LOAD MODEL (SavedModel Format)
# # ============================
# print("Loading sign language model...")
# try:
#     # Try loading SavedModel format (most compatible)
#     model = tf.keras.models.load_model('sign_model_savedmodel', compile=False)
#     print("✅ Model loaded from SavedModel format")
# except:
#     print("⚠️ SavedModel not found, trying alternative...")
#     try:
#         # Alternative: Load from weights + architecture
#         from tensorflow.keras import models, layers
        
#         # Recreate architecture
#         model = models.Sequential([
#             layers.Input(shape=(64, 64, 3)),
#             layers.Conv2D(32, (3, 3), activation='relu', padding='valid'),
#             layers.MaxPooling2D((2, 2)),
#             layers.Conv2D(64, (3, 3), activation='relu', padding='valid'),
#             layers.MaxPooling2D((2, 2)),
#             layers.Conv2D(128, (3, 3), activation='relu', padding='valid'),
#             layers.MaxPooling2D((2, 2)),
#             layers.Flatten(),
#             layers.Dense(512, activation='relu'),
#             layers.Dropout(0.5),
#             layers.Dense(35, activation='softmax')
#         ])
        
#         # Build and load weights
#         model.build((None, 64, 64, 3))
#         model.load_weights('model_weights.h5')
#         print("✅ Model loaded from weights file")
#     except Exception as e:
#         print(f"❌ Failed to load model: {e}")
#         print("\n📝 Instructions:")
#         print("1. Place 'sign_language_model.keras' in this directory")
#         print("2. Run: python convert_model.py")
#         print("3. Then run this app again")
#         exit(1)

# print(f"   Input shape: {model.input_shape}")
# print(f"   Output shape: {model.output_shape}")
# print("✅ Model ready!\n")

# ============================
# ✅ MODEL SETTINGS
# ============================
IMG_SIZE = 64

# 35 classes: 0-9 (digits) + A-Z (letters)
LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

# ============================
# ✅ IMAGE PREPROCESSING
# ============================
def preprocess_image(img):
    """Preprocess image for model prediction"""
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_sign(img):
    """Predict sign from image"""
    prediction = model.predict(img, verbose=0)[0]
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    return LABELS[class_idx], confidence

# ============================
# ✅ ISL GIF PHRASES
# ============================
isl_gif = [
    'any questions', 'are you angry', 'are you busy', 'are you hungry', 'are you sick', 'be careful',
    'can we meet tomorrow', 'did you book tickets', 'did you finish homework', 'do you go to office', 'do you have money',
    'do you want something to drink', 'do you want tea or coffee', 'do you watch TV', 'dont worry', 'flower is beautiful',
    'good afternoon', 'good evening', 'good morning', 'good night', 'good question', 'had your lunch', 'happy journey',
    'hello what is your name', 'how many people are there in your family', 'i am a clerk', 'i am bore doing nothing', 
    'i am fine', 'i am sorry', 'i am thinking', 'i am tired', 'i dont understand anything', 'i go to a theatre', 'i love to shop',
    'i had to say something but i forgot', 'i have headache', 'i like pink colour', 'i live in nagpur', 'lets go for lunch', 'my mother is a homemaker',
    'my name is john', 'nice to meet you', 'no smoking please', 'open the door', 'please call me later',
    'please clean the room', 'please give me your pen', 'please use dustbin dont throw garbage', 'please wait for sometime', 'shall I help you',
    'shall we go together tommorow', 'sign language interpreter', 'sit down', 'stand up', 'take care', 'there was traffic jam', 'wait I am thinking',
    'what are you doing', 'what is the problem', 'what is todays date', 'what is your father do', 'what is your job',
    'what is your mobile number', 'what is your name', 'whats up', 'when is your interview', 'when we will go', 'where do you stay',
    'where is the bathroom', 'where is the police station', 'you are wrong','address','agra','ahemdabad', 'all', 'april', 'assam', 'august', 'australia', 'badoda', 'banana', 'banaras', 'banglore',
    'bihar','bihar','bridge','cat', 'chandigarh', 'chennai', 'christmas', 'church', 'clinic', 'coconut', 'crocodile','dasara',
    'deaf', 'december', 'deer', 'delhi', 'dollar', 'duck', 'febuary', 'friday', 'fruits', 'glass', 'grapes', 'gujrat', 'hello',
    'hindu', 'hyderabad', 'india', 'january', 'jesus', 'job', 'july', 'july', 'karnataka', 'kerala', 'krishna', 'litre', 'mango',
    'may', 'mile', 'monday', 'mumbai', 'museum', 'muslim', 'nagpur', 'october', 'orange', 'pakistan', 'pass', 'police station',
    'post office', 'pune', 'punjab', 'rajasthan', 'ram', 'restaurant', 'saturday', 'september', 'shop', 'sleep', 'southafrica',
    'story', 'sunday', 'tamil nadu', 'temperature', 'temple', 'thursday', 'toilet', 'tomato', 'town', 'tuesday', 'usa', 'village',
    'voice', 'wednesday', 'weight','please wait for sometime','what is your mobile number','what are you doing','are you busy'
]

# ============================
# ✅ ROUTES
# ============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/listen')
def listen():
    """Speech recognition route"""
    r = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("🎤 Listening...")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)

        text = r.recognize_google(audio).lower()
        print(f"You said: {text}")
    except Exception as e:
        print(f"Speech recognition error: {e}")
        return jsonify({"type": "error", "message": "Speech not recognized"})

    # Remove punctuation
    for c in string.punctuation:
        text = text.replace(c, "")

    # Check for phrase GIF
    if text in isl_gif:
        gif_path = f"/static/ISL_Gifs/{text}.gif"
        if os.path.exists("static/ISL_Gifs/" + text + ".gif"):
            return jsonify({"type": "gif", "path": gif_path, "text": text})

    # Spell out letter by letter
    images = []
    for char in text:
        if char in string.ascii_lowercase:
            img_path = f"/static/letters/{char}.jpg"
            if os.path.exists("static/letters/" + char + ".jpg"):
                images.append(img_path)

    return jsonify({"type": "letters", "images": images, "text": text})

@app.route('/process', methods=['POST'])
def process_text():
    """Process text to sign language"""
    data = request.json
    text = data.get('text', '').lower()

    # Remove punctuation
    for c in string.punctuation:
        text = text.replace(c, "")

    # Check for phrase GIF
    if text in isl_gif:
        gif_path = f"/static/ISL_Gifs/{text}.gif"
        if os.path.exists("static/ISL_Gifs/" + text + ".gif"):
            return jsonify({"type": "gif", "path": gif_path})

    # Spell out letter by letter
    images = []
    for char in text:
        if char in string.ascii_lowercase:
            img_path = f"/static/letters/{char}.jpg"
            if os.path.exists("static/letters/" + char + ".jpg"):
                images.append(img_path)

    return jsonify({"type": "letters", "images": images})
@app.route('/text_to_sign', methods=['POST'])
def text_to_sign():
    data = request.json
    text = data.get("text", "").lower()

    for c in string.punctuation:
        text = text.replace(c, "")

    images = []
    for char in text:
        if char in string.ascii_lowercase:
            img_path = f"/static/letters/{char}.jpg"
            if os.path.exists("static/letters/" + char + ".jpg"):
                images.append(img_path)

    return jsonify({"type": "letters", "images": images, "text": text})

# ======================================
# LOAD DETR MODEL
# ======================================
print("Loading DETR model...")

transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

model = DETR(num_classes=3)
model.load_pretrained("pretrained/4426_model.pt")
model.eval()

CLASSES = get_classes()
COLORS = get_colors()

print("DETR model loaded successfully")


# ======================================
# FRAME GENERATOR WITH DETR
# ======================================
def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not accessible")
        return

    while True:
        success, frame = cap.read()
        print("Frame shape:", frame.shape)

        if not success:
            print("Frame read failed")
            break

        frame = cv2.flip(frame, 1)

        # DETR preprocessing
        transformed = transforms(image=frame)
        img_tensor = torch.unsqueeze(transformed["image"], dim=0)

        with torch.no_grad():
            output = model(img_tensor)

        probabilities = output["pred_logits"].softmax(-1)[:, :, :-1]
        max_probs, max_classes = probabilities.max(-1)
        keep = max_probs > 0.8

        batch_idx, query_idx = torch.where(keep)

        bboxes = rescale_bboxes(
            output["pred_boxes"][batch_idx, query_idx, :],
            (frame.shape[1], frame.shape[0])
        )

        classes = max_classes[batch_idx, query_idx]
        probas = max_probs[batch_idx, query_idx]

        for cls, prob, box in zip(classes, probas, bboxes):
            cls = int(cls.item())
            x1, y1, x2, y2 = box.int().tolist()

            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[cls], 3)
            label = f"{CLASSES[cls]} {prob:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

        # Encode for MJPEG stream
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n"
               + frame_bytes +
               b"\r\n")

    cap.release()




@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ============================
# ✅ RUN APP
# ============================
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)