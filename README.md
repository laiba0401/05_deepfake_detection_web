# 🔍 DeepFake Detector

An AI-powered web application for detecting deepfake images and videos using EfficientNetB0.

## 🎯 Features

- **Image Detection** — detects GAN-generated and face-swap fake images
- **Video Detection** — frame-by-frame analysis with confidence scoring
- **Detailed Results** — confidence bar, probability score, frame analysis chart
- **Real-time Analysis** — drag and drop interface with instant results

## 🧠 Model Performance

| Metric | Score |
|--------|-------|
| Overall Accuracy | 94-96% |
| Fake Detection (Recall) | 95-100% |
| Real Detection (Precision) | 96% |
| Video Accuracy | 97% (29/30) |

## 🔬 Detects

- ✅ GAN-generated faces (StyleGAN, PGGAN)
- ✅ StyleGAN2 faces (ThisPersonDoesNotExist)
- ✅ Face-swap deepfakes (DFD dataset)
- ✅ Real vs fake video classification

## 🛠️ Tech Stack

**Frontend**
- React.js
- Axios
- Recharts

**Backend**
- Django + Django REST Framework
- TensorFlow / Keras
- EfficientNetB0 (pretrained on ImageNet)
- MTCNN (face detection)
- OpenCV
## 📁 Project Structure

```
05_deepfake_detection_web/
├── backend/
│   ├── manage.py
│   ├── backend/
│   │   ├── settings.py
│   │   └── urls.py
│   └── detector/
│       ├── ml_model.py      # EfficientNet inference pipeline
│       ├── views.py         # API endpoints
│       ├── urls.py          # URL routing
│       ├── apps.py
│       └── models/          # Place keras model here (not included)
└── frontend/
    ├── package.json
    └── src/
        ├── App.js
        ├── App.css
        └── components/
            ├── ImageDetector.jsx
            ├── VideoDetector.jsx
            └── ResultCard.jsx
```
## ⚙️ Installation

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### Frontend

```bash
cd frontend
npm install
npm start
```

### Model
The trained model (`efficientnet_utkface.keras`) is not included in the repo due to file size.

**Download:** [efficientnet_utkface.keras](https://drive.google.com/file/d/12v55kIAewgBC98p8L8jjD6SQhTxlzFJM/view?usp=sharing)

Place it at:
```
backend/detector/models/efficientnet_utkface.keras
```
## 🚀 Usage

1. Start Django backend on port 8000
2. Start React frontend on port 3000
3. Open `http://localhost:3000`
4. Upload an image or video
5. Click Analyze and view results

## 📊 Training Data

- **140k Real and Fake Faces** — GAN detection
- **DFD Dataset** — face-swap video detection
- **StyleGAN2 faces** — downloaded from ThisPersonDoesNotExist
- **UTKFace Dataset** — age/diversity coverage

## 🏋️ Training Pipeline

Custom CNN (baseline)
↓
EfficientNetB0 Phase 1 (frozen base, train head)
↓
EfficientNetB0 Phase 2 (full fine-tuning)
↓
Fine-tune on DFD video frames
↓
Fine-tune on StyleGAN2 faces
↓
Fine-tune on UTKFace (age diversity)

## ⚠️ Known Limitations

- Faces with large hats or heavy occlusion may reduce accuracy
- Very young children (under 3) may have lower confidence
- Diffusion-generated images (Midjourney, Stable Diffusion) not yet supported
## 👩‍💻 Authors

- **[Laiba Tauseef](https://github.com/laiba0401)** — ML pipeline, model training, backend
- **[Syeda Maleeha Bano Naqvi](https://github.com/maleeha1d2003)** — frontend, data collection, Report Writing
