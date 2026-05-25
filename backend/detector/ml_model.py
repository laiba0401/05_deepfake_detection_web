import os
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from PIL import Image as PILImage
import tempfile

# ============================================
# THRESHOLDS
# ============================================
IMAGE_THRESHOLD = 0.56
VIDEO_THRESHOLD = 0.30
UNCERTAINTY_MIN = 0.35
UNCERTAINTY_MAX = 0.65

# ============================================
# MODEL & DETECTOR — loaded once at startup
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'efficientnet_utkface.keras')

_model = None
_mtcnn = None

def get_model():
    global _model
    if _model is None:
        print("Loading EfficientNet model...")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
    return _model

def get_detector():
    global _mtcnn
    if _mtcnn is None:
        print("Loading MTCNN detector...")
        _mtcnn = MTCNN()
        print("✅ MTCNN loaded successfully")
    return _mtcnn

# ============================================
# FACE EXTRACTION
# ============================================
def extract_face(frame_bgr, margin=40):
    detector = get_detector()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    try:
        results = detector.detect_faces(frame_rgb)
    except (ValueError, Exception):
        return None
    if not results:
        return None
    x, y, w, h = results[0]['box']
    x, y = max(0, x - margin), max(0, y - margin)
    h_img, w_img = frame_rgb.shape[:2]
    x2 = min(w_img, x + w + margin)
    y2 = min(h_img, y + h + margin)
    face = frame_rgb[y:y2, x:x2]
    if face.size == 0:
        return None
    return face

# ============================================
# IMAGE DETECTION
# ============================================
def predict_image(image_file):
    model = get_model()

    # Load image
    try:
        pil_img = PILImage.open(image_file).convert('RGB')
        img_array = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return {"error": f"Could not load image: {str(e)}"}

    # Try MTCNN face crop
    face = extract_face(img_bgr)
    face_detected = face is not None

    if face is not None:
        img_input = face
    else:
        img_input = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess
    img_resized = cv2.resize(img_input, (224, 224))
    img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(
        np.array(img_resized, dtype=np.float32)
    )

    # Predict
    prob = float(model.predict(
        np.expand_dims(img_preprocessed, axis=0), verbose=0
    )[0][0])

    verdict = "REAL" if prob > IMAGE_THRESHOLD else "FAKE"
    confidence = prob * 100 if verdict == "REAL" else (1 - prob) * 100
    uncertain = UNCERTAINTY_MIN < prob < UNCERTAINTY_MAX

    return {
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "probability": round(prob, 4),
        "face_detected": face_detected,
        "uncertain": uncertain,
        "threshold_used": IMAGE_THRESHOLD
    }

# ============================================
# VIDEO DETECTION
# ============================================
def extract_frames(video_path, sample_every_n=5):
    cap = cv2.VideoCapture(video_path)
    frames, frame_idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every_n == 0:
            frames.append(frame)
        frame_idx += 1
    cap.release()
    return frames

def predict_video(video_file):
    model = get_model()

    # Save uploaded video to temp file
    try:
        suffix = os.path.splitext(video_file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in video_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
    except Exception as e:
        return {"error": f"Could not save video: {str(e)}"}

    try:
        frames = extract_frames(tmp_path, sample_every_n=5)

        if not frames:
            return {"error": "Could not extract frames from video"}

        probs = []
        faces_found = 0
        faces_skipped = 0
        frame_results = []

        for i, frame in enumerate(frames):
            face = extract_face(frame)
            if face is None:
                faces_skipped += 1
                continue

            img_resized = cv2.resize(face, (224, 224))
            img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(
                np.array(img_resized, dtype=np.float32)
            )
            prob = float(model.predict(
                np.expand_dims(img_preprocessed, axis=0), verbose=0
            )[0][0])

            probs.append(prob)
            faces_found += 1
            frame_results.append({
                "frame": i + 1,
                "probability": round(prob, 4),
                "verdict": "REAL" if prob > VIDEO_THRESHOLD else "FAKE"
            })

        if not probs:
            return {"error": "No faces detected in video"}

        mean_prob = float(np.mean(probs))
        fake_ratio = sum(1 for p in probs if p <= VIDEO_THRESHOLD) / len(probs)
        verdict = "REAL" if mean_prob > VIDEO_THRESHOLD else "FAKE"
        confidence = mean_prob * 100 if verdict == "REAL" else (1 - mean_prob) * 100
        uncertain = UNCERTAINTY_MIN < mean_prob < UNCERTAINTY_MAX

        return {
            "verdict": verdict,
            "confidence": round(confidence, 2),
            "mean_probability": round(mean_prob, 4),
            "fake_frame_ratio": round(fake_ratio, 4),
            "fake_frame_percentage": round(fake_ratio * 100, 1),
            "faces_found": faces_found,
            "faces_skipped": faces_skipped,
            "total_frames": len(frames),
            "frame_results": frame_results,
            "uncertain": uncertain,
            "threshold_used": VIDEO_THRESHOLD
        }

    finally:
        # Always clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)