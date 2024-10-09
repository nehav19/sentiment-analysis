import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the sentiment model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'xls', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

# Route to handle single review prediction
@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    vectorized_review = vectorizer.transform([review])
    prediction = model.predict(vectorized_review)

    sentiment = 'Positive' if prediction == 1 else 'Negative'
    return jsonify(sentiment=sentiment)

# Route to handle Excel file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify(error="No selected file"), 400

    if file and allowed_file(file.filename):
        # Save the file temporarily
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        # Read the Excel file
        df = pd.read_excel(filepath)

        # Assuming the Excel file has a column named 'Review'
        reviews = df['Review'].tolist()

        # Vectorize and predict sentiment for each review
        vectorized_reviews = vectorizer.transform(reviews)
        predictions = model.predict(vectorized_reviews)

        # Calculate percentages
        positive_reviews = sum(predictions)
        negative_reviews = len(predictions) - positive_reviews

        positive_percent = (positive_reviews / len(predictions)) * 100
        negative_percent = (negative_reviews / len(predictions)) * 100

        # Remove the uploaded file after processing
        os.remove(filepath)

        # Return the percentages as a JSON response
        return jsonify(positive=positive_percent, negative=negative_percent)

    return jsonify(error="Invalid file format. Please upload an Excel file."), 400

if __name__ == '__main__':
    app.run(debug=True)
