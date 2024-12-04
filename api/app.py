import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from io import BytesIO

class VerifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = self.fc1(x)
        return x

# Define the image preprocessing and prediction functions
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5647, 0.4770, 0.4273), (0.2724, 0.2619, 0.2676))
    ])
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict_image(image_bytes, model):
    image = preprocess_image(image_bytes)
    with torch.no_grad():
        outputs = model(image)
        softmax_probs = torch.softmax(outputs, dim=1)
        percent_ai = softmax_probs[0][0].item()
        return percent_ai

# Initialize Flask app
app = Flask(__name__)
cors = CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})  # Allow CORS only for the specific route
app.config['CORS_HEADERS'] = 'Content-Type'

# Initialize the model
model = VerifierCNN()

# Load the saved model weights
try:
    model.load_state_dict(torch.load('cnn_weights.pth', map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: cnn_weights.pth file not found.")
    exit(1)

# Prediction route
@app.route('/predict', methods=['POST'])
@cross_origin()  # Enable CORS for this specific route
def predict():
    print("Somebody called me.")
    if 'image' not in request.files:
        return jsonify({'error': 'No image part provided'}), 400

    image = request.files['image']

    try:
        percent_ai = predict_image(image.read(), model)
        return jsonify({'percent_ai': percent_ai})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
