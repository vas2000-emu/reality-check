from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Define the CNN model
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
        # Flatten dynamically
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return F.softmax(x, dim=1)

# Initialize Flask app
app = Flask(__name__)

# Initialize the model
model = VerifierCNN()

# Load the saved model weights
try:
    model.load_state_dict(torch.load('final_cnn_model.pth', map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: final_cnn_model.pth file not found.")
    exit(1)

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5647, 0.4770, 0.4273), (0.2724, 0.2619, 0.2676))
])

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part provided'}), 400

    image = request.files['image']

    try:
        img = Image.open(BytesIO(image.read())).convert('RGB')
        img = transform(img).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            prediction = "AI-Generated" if predicted.item() == 0 else "Real"

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
