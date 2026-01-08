from flask import Flask, render_template, jsonify, Response, request
import cv2
import numpy as np
import tensorflow as tf
import os
import string
import speech_recognition as sr
from flask import request

# NLP Libraries
import spacy
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

app = Flask(__name__)

# ============================
# ✅ INITIALIZE NLP MODELS
# ============================
print("Initializing NLP models...")

try:
    # Load SpaCy model for advanced NLP
    nlp_spacy = spacy.load("en_core_web_sm")
    print("✅ SpaCy model loaded")
except:
    print("⚠️ SpaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp_spacy = None

try:
    # Download required NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print("✅ NLTK data downloaded")
except:
    print("⚠️ NLTK download failed")

try:
    # Initialize sentiment analyzer
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    print("✅ Sentiment analyzer loaded")
except:
    print("⚠️ Sentiment analyzer not available")
    sentiment_analyzer = None

try:
    # Initialize summarizer
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("✅ Summarizer loaded")
except:
    print("⚠️ Summarizer not available")
    summarizer = None

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# ============================
# ✅ LOAD SIGN LANGUAGE MODEL
# ============================
print("\nLoading sign language model...")
try:
    model = tf.keras.models.load_model('sign_model_savedmodel', compile=False)
    print("✅ Model loaded from SavedModel format")
except:
    print("⚠️ SavedModel not found, trying alternative...")
    try:
        from tensorflow.keras import models, layers
        
        model = models.Sequential([
            layers.Input(shape=(64, 64, 3)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='valid'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='valid'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='valid'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(35, activation='softmax')
        ])
        
        model.build((None, 64, 64, 3))
        model.load_weights('model_weights.h5')
        print("✅ Model loaded from weights file")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        exit(1)

print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")
print("✅ Model ready!\n")

# ============================
# ✅ MODEL SETTINGS
# ============================
IMG_SIZE = 64

LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

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
    'bihar','bridge','cat', 'chandigarh', 'chennai', 'christmas', 'church', 'clinic', 'coconut', 'crocodile','dasara',
    'deaf', 'december', 'deer', 'delhi', 'dollar', 'duck', 'febuary', 'friday', 'fruits', 'glass', 'grapes', 'gujrat', 'hello',
    'hindu', 'hyderabad', 'india', 'january', 'jesus', 'job', 'july', 'karnataka', 'kerala', 'krishna', 'litre', 'mango',
    'may', 'mile', 'monday', 'mumbai', 'museum', 'muslim', 'nagpur', 'october', 'orange', 'pakistan', 'pass', 'police station',
    'post office', 'pune', 'punjab', 'rajasthan', 'ram', 'restaurant', 'saturday', 'september', 'shop', 'sleep', 'southafrica',
    'story', 'sunday', 'tamil nadu', 'temperature', 'temple', 'thursday', 'toilet', 'tomato', 'town', 'tuesday', 'usa', 'village',
    'voice', 'wednesday', 'weight','please wait for sometime','what is your mobile number','what are you doing','are you busy'
]

# ============================
# ✅ NLP HELPER FUNCTIONS
# ============================

def preprocess_text_nlp(text):
    """Advanced text preprocessing using NLP"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_keywords(text):
    """Extract important keywords using NLP"""
    if nlp_spacy:
        doc = nlp_spacy(text)
        keywords = []
        
        # Extract nouns, verbs, and adjectives
        for token in doc:
            if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                keywords.append(token.lemma_)
        
        return list(set(keywords))
    else:
        # Fallback: simple tokenization
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        keywords = [w for w in words if w.isalnum() and w not in stop_words]
        return keywords

def get_entities(text):
    """Extract named entities from text"""
    if nlp_spacy:
        doc = nlp_spacy(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    return []

def simplify_sentence(text):
    """Simplify complex sentences for better ISL translation"""
    if nlp_spacy:
        doc = nlp_spacy(text)
        
        # Extract main subject-verb-object structure
        simplified_parts = []
        for sent in doc.sents:
            # Get root verb
            root = sent.root
            subject = None
            obj = None
            
            # Find subject and object
            for child in root.children:
                if child.dep_ in ['nsubj', 'nsubjpass']:
                    subject = child.text
                elif child.dep_ in ['dobj', 'pobj']:
                    obj = child.text
            
            # Build simplified sentence
            if subject and obj:
                simplified_parts.append(f"{subject} {root.lemma_} {obj}")
            elif subject:
                simplified_parts.append(f"{subject} {root.lemma_}")
        
        return " ".join(simplified_parts) if simplified_parts else text
    
    return text

def get_sentiment(text):
    """Analyze sentiment of the text"""
    if sentiment_analyzer:
        try:
            result = sentiment_analyzer(text[:512])[0]  # Limit to 512 chars
            return {
                'label': result['label'],
                'score': result['score']
            }
        except:
            return None
    return None

def chunk_long_text(text, max_length=50):
    """Break long text into manageable chunks"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        words = word_tokenize(sentence)
        if current_length + len(words) <= max_length:
            current_chunk.append(sentence)
            current_length += len(words)
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(words)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def match_phrase_with_nlp(text):
    """Intelligently match text to available ISL GIF phrases using NLP"""
    text_clean = preprocess_text_nlp(text)
    
    # Direct match
    if text_clean in isl_gif:
        return text_clean
    
    # Try lemmatization match
    if nlp_spacy:
        doc = nlp_spacy(text_clean)
        lemmatized = ' '.join([token.lemma_ for token in doc])
        if lemmatized in isl_gif:
            return lemmatized
    
    # Try fuzzy matching based on keywords
    text_keywords = set(extract_keywords(text_clean))
    best_match = None
    best_score = 0
    
    for phrase in isl_gif:
        phrase_keywords = set(extract_keywords(phrase))
        intersection = text_keywords.intersection(phrase_keywords)
        
        if len(intersection) > 0:
            score = len(intersection) / max(len(text_keywords), len(phrase_keywords))
            if score > best_score and score > 0.6:  # 60% match threshold
                best_score = score
                best_match = phrase
    
    return best_match

def translate_to_isl_structure(text):
    """
    Translate English text to ISL grammatical structure
    ISL typically follows: Time + Topic + Comment + Question
    and doesn't use articles, prepositions as much
    """
    if not nlp_spacy:
        return text
    
    doc = nlp_spacy(text)
    isl_words = []
    
    for token in doc:
        # Skip articles, auxiliary verbs, and some prepositions
        if token.pos_ in ['DET', 'AUX'] or token.dep_ in ['aux', 'auxpass']:
            continue
        
        # Add content words
        if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN', 'NUM', 'ADV']:
            # Use base form for verbs
            if token.pos_ == 'VERB':
                isl_words.append(token.lemma_)
            else:
                isl_words.append(token.text.lower())
        
        # Keep question words
        elif token.tag_ in ['WP', 'WRB', 'WDT']:
            isl_words.append(token.text.lower())
    
    return ' '.join(isl_words)

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
# ✅ ROUTES
# ============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    """Analyze text using NLP before translation"""
    data = request.json
    text = data.get('text', '')
    
    # Preprocess
    processed_text = preprocess_text_nlp(text)
    
    # Get NLP analysis
    analysis = {
        'original': text,
        'processed': processed_text,
        'keywords': extract_keywords(processed_text),
        'entities': get_entities(processed_text),
        'sentiment': get_sentiment(processed_text),
        'simplified': simplify_sentence(processed_text),
        'isl_structure': translate_to_isl_structure(processed_text),
        'chunks': chunk_long_text(processed_text)
    }
    
    return jsonify(analysis)

@app.route('/listen')
def listen():
    """Speech recognition route with NLP enhancement"""
    r = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("🎤 Listening...")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)

        text = r.recognize_google(audio)
        print(f"You said: {text}")
    except Exception as e:
        print(f"Speech recognition error: {e}")
        return jsonify({"type": "error", "message": "Speech not recognized"})

    # NLP preprocessing
    processed_text = preprocess_text_nlp(text)
    
    # Get sentiment
    sentiment = get_sentiment(processed_text)
    
    # Try to match with ISL phrases using NLP
    matched_phrase = match_phrase_with_nlp(processed_text)
    
    if matched_phrase:
        gif_path = f"/static/ISL_Gifs/{matched_phrase}.gif"
        if os.path.exists("static/ISL_Gifs/" + matched_phrase + ".gif"):
            return jsonify({
                "type": "gif",
                "path": gif_path,
                "text": text,
                "matched_phrase": matched_phrase,
                "sentiment": sentiment,
                "keywords": extract_keywords(processed_text)
            })
    
    # Convert to ISL structure
    isl_text = translate_to_isl_structure(processed_text)
    
    # Spell out letter by letter
    images = []
    for char in isl_text:
        if char in string.ascii_lowercase:
            img_path = f"/static/letters/{char}.jpg"
            if os.path.exists("static/letters/" + char + ".jpg"):
                images.append(img_path)
        elif char == ' ':
            images.append("/static/letters/space.jpg")  # Add space indicator if available

    return jsonify({
        "type": "letters",
        "images": images,
        "text": text,
        "isl_text": isl_text,
        "sentiment": sentiment,
        "keywords": extract_keywords(processed_text)
    })

@app.route('/process', methods=['POST'])
def process_text():
    """Process text to sign language with NLP enhancement"""
    data = request.json
    text = data.get('text', '')

    # NLP preprocessing
    processed_text = preprocess_text_nlp(text)
    
    # Try to match with ISL phrases
    matched_phrase = match_phrase_with_nlp(processed_text)
    
    if matched_phrase:
        gif_path = f"/static/ISL_Gifs/{matched_phrase}.gif"
        if os.path.exists("static/ISL_Gifs/" + matched_phrase + ".gif"):
            return jsonify({
                "type": "gif",
                "path": gif_path,
                "matched_phrase": matched_phrase,
                "keywords": extract_keywords(processed_text)
            })

    # Convert to ISL structure
    isl_text = translate_to_isl_structure(processed_text)
    
    # Spell out letter by letter
    images = []
    for char in isl_text:
        if char in string.ascii_lowercase:
            img_path = f"/static/letters/{char}.jpg"
            if os.path.exists("static/letters/" + char + ".jpg"):
                images.append(img_path)

    return jsonify({
        "type": "letters",
        "images": images,
        "isl_text": isl_text,
        "original_text": text
    })

@app.route('/text_to_sign', methods=['POST'])
def text_to_sign():
    """Convert text to sign language with full NLP pipeline"""
    data = request.json
    text = data.get("text", "")

    # NLP Analysis
    processed_text = preprocess_text_nlp(text)
    keywords = extract_keywords(processed_text)
    entities = get_entities(processed_text)
    sentiment = get_sentiment(processed_text)
    
    # For long text, chunk it
    chunks = chunk_long_text(processed_text)
    
    all_images = []
    chunk_info = []
    
    for chunk in chunks:
        # Try phrase matching first
        matched_phrase = match_phrase_with_nlp(chunk)
        
        if matched_phrase and os.path.exists("static/ISL_Gifs/" + matched_phrase + ".gif"):
            chunk_info.append({
                "type": "gif",
                "path": f"/static/ISL_Gifs/{matched_phrase}.gif",
                "text": chunk
            })
        else:
            # Convert to ISL structure
            isl_text = translate_to_isl_structure(chunk)
            
            images = []
            for char in isl_text:
                if char in string.ascii_lowercase:
                    img_path = f"/static/letters/{char}.jpg"
                    if os.path.exists("static/letters/" + char + ".jpg"):
                        images.append(img_path)
            
            chunk_info.append({
                "type": "letters",
                "images": images,
                "text": chunk,
                "isl_text": isl_text
            })
    
    return jsonify({
        "chunks": chunk_info,
        "analysis": {
            "keywords": keywords,
            "entities": entities,
            "sentiment": sentiment,
            "original_text": text
        }
    })

@app.route('/smart_translate', methods=['POST'])
def smart_translate():
    """
    Smart translation endpoint that uses full NLP pipeline
    for context-aware ISL translation
    """
    data = request.json
    text = data.get("text", "")
    context = data.get("context", "")  # Optional context from previous messages
    
    # Combine with context if available
    full_text = f"{context} {text}" if context else text
    
    # NLP Analysis
    processed_text = preprocess_text_nlp(text)
    simplified = simplify_sentence(processed_text)
    isl_structure = translate_to_isl_structure(simplified)
    
    # Get analysis
    analysis = {
        "keywords": extract_keywords(processed_text),
        "entities": get_entities(processed_text),
        "sentiment": get_sentiment(processed_text),
        "original": text,
        "simplified": simplified,
        "isl_structure": isl_structure
    }
    
    # Try phrase matching
    matched_phrase = match_phrase_with_nlp(isl_structure)
    
    if matched_phrase and os.path.exists("static/ISL_Gifs/" + matched_phrase + ".gif"):
        return jsonify({
            "type": "phrase",
            "gif_path": f"/static/ISL_Gifs/{matched_phrase}.gif",
            "matched_phrase": matched_phrase,
            "analysis": analysis
        })
    
    # Letter-by-letter translation
    images = []
    for char in isl_structure:
        if char in string.ascii_lowercase:
            img_path = f"/static/letters/{char}.jpg"
            if os.path.exists("static/letters/" + char + ".jpg"):
                images.append(img_path)
    
    return jsonify({
        "type": "letters",
        "images": images,
        "analysis": analysis
    })

# ============================
# ✅ LIVE CAMERA STREAM
# ============================
def generate_frames():
    """Generate frames from webcam"""
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Center ROI
        roi_size = 300
        cx, cy = w // 2, h // 2
        x1, y1 = cx - roi_size // 2, cy - roi_size // 2
        x2, y2 = cx + roi_size // 2, cy + roi_size // 2
        roi = frame[y1:y2, x1:x2]

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        if roi.size > 0:
            # Predict
            input_img = preprocess_image(roi)
            label, confidence = predict_sign(input_img)
            
            text = f"{label} ({confidence:.2f})"
            cv2.putText(frame, text, (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Encode frame
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ============================
# ✅ RUN APP
# ============================
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)