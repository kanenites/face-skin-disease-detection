from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('DATA/disease_classification_model.h5')

class_labels = {0: 'Acne', 1: 'Eczema', 2: 'Actinic Keratosis', 3: 'Basal Cell Carcinoma', 4: 'Rosacea' }  # Update with your class indices

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    filepath = os.path.join('Uploads', file.filename)
    file.save(filepath)
    
    img_array = prepare_image(filepath)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    disease = class_labels[predicted_class]
    
    return jsonify({'disease': disease})

if __name__ == '__main__':
    app.run(debug=True)
