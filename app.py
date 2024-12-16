from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load model and preprocessing data
model = tf.keras.models.load_model('char_model.keras')
mean_net = np.loadtxt('mean.txt', dtype=float).reshape(1, 28 * 28)
std_net = np.loadtxt('standy.txt', dtype=float).reshape(1, 28 * 28)
labels = pd.read_csv('Datasets/emnist-bymerge-mapping.txt', delimiter=' ', header=None, index_col=0).squeeze()

label_dict = {index: chr(label) for index, label in enumerate(labels)}


def preprocess(img):
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(1, 28 * 28)
    img = (img - mean_net) / std_net
    img = img.reshape(1, 28, 28, 1)
    pred = np.argmax(model.predict(img))
    return label_dict[pred]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get image data from request
    image_data = request.form['image_data']
    image_data = image_data.split(',')[1]  # Remove header
    image_data = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_data)).convert('RGB')
    image = np.array(image)

    # Preprocess and predict
    prediction = preprocess(image)
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)
