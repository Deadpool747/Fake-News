from flask import Flask, render_template, request
from PIL import Image
import pytesseract
import pickle
import os

# Load the trained model and vectorizer
model_path = 'model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Model or vectorizer file not found. Train the model first.")

with open(model_path, 'rb') as file:
    model = pickle.load(file)
with open(vectorizer_path, 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to process text
def process_text(text):
    """Preprocess the input text."""
    return text.strip()

# Function to process an image and extract text
def process_image(image_path):
    """Extract text from an image using pytesseract."""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"Error processing image: {e}"

# Predict function
def predict_news(input_text=None, image_path=None):
    """Predict if the news is fake or true based on text or an image."""
    if input_text:
        text = process_text(input_text)
    elif image_path:
        text = process_image(image_path)
    else:
        return "No valid input provided."
    
    # Vectorize the input text
    vectorized_text = tfidf_vectorizer.transform([text])
    
    # Predict using the model
    prediction = model.predict(vectorized_text)[0]
    prediction_label = "True News" if prediction == 1 else "Fake News"
    
    return prediction_label

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # If text is submitted
        input_text = request.form.get("text_input")
        if input_text:
            result = predict_news(input_text=input_text)
            return render_template("index.html", prediction=result)

        # If an image is uploaded
        uploaded_file = request.files.get("image_input")
        if uploaded_file and uploaded_file.filename != '':
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(file_path)
            result = predict_news(image_path=file_path)
            return render_template("index.html", prediction=result)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

