from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Define the model using only ShuffleNetV2
class ShuffleNetV2Model(nn.Module):
    def __init__(self, num_classes=5):  # Ensure this matches the saved model
        super(ShuffleNetV2Model, self).__init__()
        self.shufflenet = models.shufflenet_v2_x1_5(weights=None)  # Load without pre-trained weights
        self.shufflenet.fc = nn.Sequential(
            nn.Linear(self.shufflenet.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)  # Ensure this matches the saved model
        )

    def forward(self, x):
        return self.shufflenet(x)

# Load the trained model
try:
    model = ShuffleNetV2Model(num_classes=5)  # Ensure this matches the saved model
    model.load_state_dict(torch.load('shufflenetv2_plant_disease_model.pth', map_location=torch.device('cpu')))
    model.eval()
    app.logger.info("Model loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading model: {e}")

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels and details (replace with your actual data)
class_labels = ['Healthy', 'Soyabean Semilooper_Pest_Attack', 'Soyabean_Mosaic', 'Soyabean_Rust']
class_details = {
    'Healthy': {'symptoms': 'No symptoms detected.', 'treatment': 'No treatment required.'},
    'Soyabean Semilooper_Pest_Attack': {'symptoms': 'Chewed leaves, defoliation, and larvae visible on plants.', 'treatment': 'Use insecticides like chlorantraniliprole or neem oil; encourage natural predators.'},
    'Soyabean_Mosaic': {'symptoms': 'Mottled yellow-green leaves, stunted growth, and distorted pods.', 'treatment': 'No cure; use virus-free seeds, control aphids, and remove infected plants.'},
    'Soyabean_Rust': {'symptoms': 'Yellow-orange pustules on leaves, premature leaf drop.', 'treatment': 'Apply fungicides like azoxystrobin; plant resistant varieties and ensure crop rotation.'},
}

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            app.logger.error("No file uploaded")
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            app.logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400

        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = class_labels[predicted.item()]
            confidence_score = confidence.item()

        # Get additional details
        details = class_details.get(predicted_class, {'symptoms': 'Unknown', 'treatment': 'Unknown'})

        app.logger.info(f"Prediction: {predicted_class}, Confidence: {confidence_score}")
        return jsonify({
            'prediction': predicted_class,
            'confidence': round(confidence_score * 100, 2),  # Convert to percentage
            'symptoms': details['symptoms'],
            'treatment': details['treatment']
        })
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True)