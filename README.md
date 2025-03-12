# **Handwritten Character Recognizer**

## **📌 Overview**  
The **Handwritten Character Recognizer** is a Flask-based web application that uses a deep learning model to recognize handwritten characters. Users can draw a character on a web interface, and the application processes and predicts the character using a **Convolutional Neural Network (CNN)** trained on the **EMNIST dataset**.  

### **🛠️ How It Works**  
1. Users draw a character on the webpage.  
2. The drawing is sent to the backend as a **base64 image**.  
3. The backend **preprocesses the image** (resizes, converts to grayscale, normalizes).  
4. The processed image is fed into a **trained CNN model**.  
5. The model **predicts the character** and returns it to the frontend.  

---

## **🌟 Features**  
✅ **Handwritten Character Recognition** – Supports letters and digits.  
✅ **Flask API** – Web-based interface to interact with the model.  
✅ **Deep Learning with TensorFlow** – Trained on the **EMNIST dataset**.  
✅ **Real-time Predictions** – Users can draw and instantly receive results.  
✅ **Preprocessing Pipeline** – Image resizing, grayscale conversion, normalization.  

---

## **📁 Project Structure**
```
HandWritten-Character-Recognizer/
│── Dockerfile
│── LICENSE
│── Model_Creation.py.ipynb
│── README.md
│── Recognition.ipynb
│── Scaler_Features.py
│── app.py
│── char_model.keras
│── classifier.py
│── mean.txt
│── requirements.txt
│── standy.txt
└── templates/
    └── index.html
```
- **`app.py`** – Main Flask application.  
- **`char_model.keras`** – Pre-trained deep learning model.  
- **`mean.txt` & `standy.txt`** – Used for input normalization.  
- **`index.html`** – Webpage for drawing characters.  
- **`requirements.txt`** – Lists all dependencies.  

---

## **🚀 Getting Started**
### **📌 Prerequisites**  
Before running the project, ensure you have the following:  
- 🔹 Python 3.8+  
- 🔹 pip (Python package manager)  
- 🔹 Flask  
- 🔹 TensorFlow  
- 🔹 OpenCV  
- 🔹 NumPy & Pandas  

### **📥 Installation**
#### **🔹 Clone the repository**  
```sh
git clone https://github.com/Zxenith/HandWritten-Digit-Recognizer
cd HandWritten-Digit-Recognizer
```
#### **🔹 Install dependencies**  
```sh
pip install -r requirements.txt
```
#### **🔹 Run the Flask app**  
```sh
python app.py
```
The app will be available at `http://127.0.0.1:5000/`

---

## **📌 Usage**  
1. Open the web browser and go to `http://127.0.0.1:5000/`  
2. Draw a character using the drawing pad.  
3. Click the **Predict** button.  
4. The model will process the input and display the recognized character.  

---

## **🛠️ Testing**  
To test the model’s accuracy with sample images:  
```sh
pytest tests.py
```

---

## **📌 Project Roadmap**  
✅ **Stage 1:** Basic recognition model  
✅ **Stage 2:** Flask API integration  
🔜 **Stage 3:** Improve accuracy with more training data  
🔜 **Stage 4:** Deploy online  

---

## **🤝 Contributing**  
Want to improve the project? Follow these steps:  
1. **Fork the repository**  
2. **Clone it locally**  
   ```sh
   git clone https://github.com/Zxenith/HandWritten-Digit-Recognizer
   ```
3. **Create a feature branch**  
   ```sh
   git checkout -b new-feature
   ```
4. **Commit your changes**  
   ```sh
   git commit -m "Added new feature"
   ```
5. **Push and open a pull request**  

---

## **📜 License**  
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.  

---

## **🙏 Acknowledgments**  
🔹 **TensorFlow/Keras** – For building the deep learning model.  
🔹 **Flask** – For creating the API.  
🔹 **OpenCV & NumPy** – For image processing.  

---
