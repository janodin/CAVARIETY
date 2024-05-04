import base64
import cv2
import numpy as np
from skimage.feature import hog
from django.shortcuts import render
from app.forms import ImageUploadForm
import joblib

# Load the trained SVM model
model = joblib.load('static/svm_model.joblib')

# List of class names your model can predict
varieties = ['BR25', 'K-1', 'PBC_123']


def extract_hog_features(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize images to ensure uniformity
    resized_img = cv2.resize(gray_image, (128, 128))
    # Extract HOG features
    fd, _ = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=True)
    return fd


def index(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Read the uploaded image
                uploaded_image = form.cleaned_data['image']
                image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

                # Preprocess the image and extract features
                extracted_features = extract_hog_features(image)

                # Predict the probability and class
                proba = model.predict_proba([extracted_features])[0]
                prediction = model.predict([extracted_features])[0]

                # Map the numeric prediction back to variety name
                predicted_label = varieties[prediction]  # Adjust index if necessary
                max_probability = np.max(proba)
                accuracy = max_probability * 100  # Convert to percentage

                if accuracy < 85:
                    # Load no_image_found.jpg
                    with open('static/not_detected.png', 'rb') as image_file:
                        no_image_data = base64.b64encode(image_file.read()).decode('utf-8')
                    image_data = f'data:image/jpeg;base64,{no_image_data}'

                    return render(request, 'result.html', {
                        'image_data': image_data,
                        'predicted_label': "Unknown Image, please try again!",
                        'accuracy': 'N/A'
                    })

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
            except Exception as e:
                form.add_error(None, f"Error processing image: {str(e)}")
    else:
        form = ImageUploadForm()

    return render(request, 'index.html', {'form': form})
