from flask import Flask, render_template, request
from transformers import pipeline
from fer import FER
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load sentiment analysis pipeline (text)
sentiment_pipeline = pipeline("sentiment-analysis")

# Initialize FER detector (image emotion)
detector = FER(mtcnn=True)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    image_result = ""

    if request.method == "POST":
        # Handle text input
        text = request.form.get("text", "")
        if text.strip():
            result = sentiment_pipeline(text)[0]
            prediction = f"{result['label']} (score: {result['score']:.2f})"

        # Handle image input
        image_file = request.files.get("image")
        if image_file and image_file.filename != "":
            # Read image bytes and convert to OpenCV format
            image_bytes = image_file.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            open_cv_image = np.array(pil_image)
            open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR

            # Detect emotions
            emotions = detector.detect_emotions(open_cv_image)
            if emotions:
                first_face = emotions[0]
                scores = first_face["emotions"]
                top_emotion = max(scores, key=scores.get)
                confidence = scores[top_emotion]
                image_result = f"{top_emotion.capitalize()} (confidence: {confidence:.2f})"
            else:
                image_result = "No face detected"

    return render_template("index.html", prediction=prediction, image_result=image_result)

if __name__ == "__main__":
    app.run(debug=True)
