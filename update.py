from flask import Flask, render_template, jsonify, Response, request
import cv2
import os
import string
import speech_recognition as sr
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import spacy

# DETR imports
from model import DETR
from utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors

# ============================
# APP INIT
# ============================
app = Flask(__name__)

# ============================
# NLP SETUP
# ============================
nlp = spacy.load("en_core_web_sm")

AUX_VERBS = {
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had"
}
ARTICLES = {"a", "an", "the"}
POLITE_WORDS = {"please", "kindly"}

def find_phrase_gif(text):
    """
    Finds the longest matching ISL GIF phrase from static/ISL_Gifs
    """
    gif_dir = "static/ISL_Gifs"
    if not os.path.exists(gif_dir):
        return None

    text = text.strip().lower()

    # Check longest phrases first
    words = text.split()
    for length in range(len(words), 0, -1):
        phrase = " ".join(words[:length])
        gif_path = f"{gif_dir}/{phrase}.gif"
        if os.path.exists(gif_path):
            return f"/static/ISL_Gifs/{phrase}.gif"

    return None

def isl_sov_transform(sentence):
    doc = nlp(sentence)

    subject = []
    obj = []
    verb = []
    adj = []

    for token in doc:
        word = token.text.lower()

        if word in AUX_VERBS or word in ARTICLES or word in POLITE_WORDS:
            continue

        if token.dep_ in ("nsubj", "nsubjpass"):
            subject.append(word)

        elif token.dep_ in ("dobj", "pobj", "obj"):
            obj.append(word)

        elif token.pos_ == "VERB":
            verb.append(token.lemma_)

        elif token.pos_ == "ADJ":
            adj.append(word)

    # Case 1: Verb exists → use SOV
    if verb:
        return " ".join(subject + obj + verb + adj)

    # Case 2: No verb (questions / states like "you busy")
    if subject and adj:
        return " ".join(subject + adj)

    # Fallback
    return sentence

# ============================
# ISL GIF PHRASES
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
# HOME
# ============================
@app.route('/')
def index():
    return render_template('index.html')

# ============================
# SPEECH TO ISL
# ============================
@app.route('/listen')
def listen():
    r = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)

        text = r.recognize_google(audio).lower()

    except Exception:
        return jsonify({
            "type": "error",
            "message": "Speech not recognized"
        })

    # Remove punctuation
    for c in string.punctuation:
        text = text.replace(c, "")

    # =========================
    # PHRASE-LEVEL GIF PRIORITY
    # =========================
    raw_text = text

    phrase_gif = find_phrase_gif(raw_text)
    if phrase_gif:
        return jsonify({
            "type": "gif",
            "path": phrase_gif,
            "text": raw_text
        })

    # =========================
    # ISL GRAMMAR TRANSFORM
    # =========================
    text = isl_sov_transform(raw_text)

    # =========================
    # WORD → LETTER SEQUENCE
    # =========================
    images = []
    words = text.split()

    for word in words:
        word_gif_path = f"/static/ISL_Gifs/{word}.gif"
        if os.path.exists("static/ISL_Gifs/" + word + ".gif"):
            images.append(word_gif_path)
            continue

        for char in word:
            if char in string.ascii_lowercase:
                letter_path = f"/static/letters/{char}.jpg"
                if os.path.exists("static/letters/" + char + ".jpg"):
                    images.append(letter_path)

    return jsonify({
        "type": "sequence",
        "images": images,
        "text": text
    })


# ============================
# TEXT TO ISL
# ============================
@app.route('/text_to_sign', methods=['POST'])
def text_to_sign():
    data = request.json
    raw_text = data.get("text", "").lower()

    for c in string.punctuation:
        raw_text = raw_text.replace(c, "")

    # 🔴 PHRASE GIF PRIORITY
    phrase_gif = find_phrase_gif(raw_text)
    if phrase_gif:
        return jsonify({
            "type": "gif",
            "path": phrase_gif,
            "text": raw_text
        })

    # Apply ISL grammar only if no phrase GIF
    text = isl_sov_transform(raw_text)

    images = []
    for word in text.split():
        word_gif = f"/static/ISL_Gifs/{word}.gif"
        if os.path.exists("static/ISL_Gifs/" + word + ".gif"):
            images.append(word_gif)
            continue

        for char in word:
            if char in string.ascii_lowercase:
                letter_path = f"/static/letters/{char}.jpg"
                if os.path.exists("static/letters/" + char + ".jpg"):
                    images.append(letter_path)

    return jsonify({
        "type": "sequence",
        "images": images,
        "text": text
    })


# ============================
# DETR MODEL LOAD
# ============================
print("Loading DETR model...")

transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

model = DETR(num_classes=3)
model.load_pretrained("pretrained/4426_model.pt")
model.eval()

CLASSES = get_classes()
COLORS = get_colors()

print("DETR loaded")

# ============================
# VIDEO STREAM
# ============================
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        transformed = transforms(image=frame)
        img_tensor = transformed["image"].unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)

        probs = output["pred_logits"].softmax(-1)[:, :, :-1]
        max_probs, classes = probs.max(-1)
        keep = max_probs > 0.8

        idx = torch.where(keep)
        boxes = rescale_bboxes(
            output["pred_boxes"][idx],
            (frame.shape[1], frame.shape[0])
        )

        for cls, prob, box in zip(classes[idx], max_probs[idx], boxes):
            x1, y1, x2, y2 = box.int().tolist()
            label = f"{CLASSES[int(cls)]} {prob:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[int(cls)], 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

    cap.release()

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ============================
# RUN
# ============================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
