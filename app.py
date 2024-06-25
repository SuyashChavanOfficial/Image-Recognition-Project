from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model('final_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((224, 224))  # Resize to the input shape of your model
    img_array = np.expand_dims(img, axis=0) / 255.0  # Preprocess image
    prediction = model.predict(img_array).tolist()
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
