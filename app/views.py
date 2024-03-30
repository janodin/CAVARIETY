import base64

import cv2
import numpy as np
from django.http import HttpResponse
from django.shortcuts import render

from app.forms import ImageUploadForm
import joblib

# Load the trained SVM model
model = joblib.load('static/svm_model.joblib')

# List of class names your model can predict
varieties = ['BR25', 'K-1', 'PBC_123']


def index(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Read the uploaded image
            uploaded_image = form.cleaned_data['image']
            image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

            # Preprocess the image (similar to how you did in training)
            resized = cv2.resize(image, (200, 200))
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            means, stds = cv2.meanStdDev(hsv)
            color_features = np.concatenate([means, stds]).flatten()

            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            shape_features = cv2.Canny(blurred, 200, 300).flatten()

            combined_features = np.concatenate([color_features, shape_features]).flatten()
            reshaped = combined_features.reshape(1, -1)

            # Predict the class probabilities using the SVM model
            probabilities = model.predict_proba(reshaped)[0]
            predicted_label_index = np.argmax(probabilities)
            predicted_label = varieties[predicted_label_index]
            accuracy = probabilities[predicted_label_index] * 100  # Convert to percentage

            # Convert the image to Base64 for displaying on the web
            _, buffer = cv2.imencode('.jpg', image)
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            image_data = f'data:image/jpeg;base64,{encoded_image}'

            # Prepare the response
            context = {
                'image_data': image_data,
                'predicted_label': predicted_label,
                'accuracy': f'{accuracy:.2f}%'
            }
            return render(request, 'result.html', context)
    else:
        form = ImageUploadForm()

    return render(request, 'index.html', {'form': form})
