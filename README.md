# Face Skin Disease Detection

A Flask web app that classifies facial skin diseases from uploaded images using transfer learning with VGG16. Built as an Honours AI/ML project at Xavier Institute of Engineering, Mumbai.

## Detects 5 Conditions

- Acne
- Eczema
- Actinic Keratosis
- Basal Cell Carcinoma
- Rosacea

## How It Works

1. User uploads a skin image via the web interface
2. Flask saves the image to `Uploads/`
3. VGG16-based model (fine-tuned on custom dataset) classifies the image
4. Predicted disease is returned as a JSON response

## Tech Stack

Python · Flask · TensorFlow / Keras · VGG16 (transfer learning) · NumPy

## Project Structure

```
AIML/
├── app.py                              # Flask web application
├── model_training.py                   # VGG16 transfer learning training script
├── templates/
│   └── index.html                      # Upload UI
├── DATA/
│   ├── train/                          # Training images (5 classes, 88 images each)
│   └── testing/                        # Test images
└── Uploads/                            # Runtime upload folder
```

## Installation

```bash
pip install flask tensorflow numpy pillow
```

## Running the App

> **Important:** The trained model file (`DATA/disease_classification_model.h5`) is not included in this repository due to file size constraints. You must train it first:

```bash
# Step 1 — Train the model (requires the DATA/ folder with images)
python model_training.py

# Step 2 — Run the Flask app (once model is trained and saved)
python app.py
```

Then open `http://localhost:5000` and upload a skin image.

## Training the Model

`model_training.py` fine-tunes VGG16 (ImageNet weights) with:
- Image augmentation (rotation, shift, shear, zoom, flip)
- EarlyStopping and ReduceLROnPlateau callbacks
- Adam optimizer (lr=0.0001)
- 50 max epochs

The trained model is saved as `DATA/disease_classification_model.h5`.

---
*Kunal's Lab — AI & ML Systems · For research and educational purposes only.*
