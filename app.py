import base64
import io
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms

# --- PyTorch Model Definition ---
# This class definition MUST be identical to the one used during model training
# to ensure the pickled model loads correctly.
class SimpleCNN(nn.Module):
    """A simple Convolutional Neural Network."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- Global Variables & Initialization ---
app = Flask(__name__)

# Load the trained PyTorch model
# Ensure 'eurosat_model.pkl' is in the same directory as this script.
try:
    model = torch.load("eurosat_model.pkl", map_location=torch.device('cpu'), weights_only=False)
    model.eval()  # Set the model to evaluation mode
    print("PyTorch model loaded successfully.")
except FileNotFoundError:
    print("Error: 'eurosat_model.pkl' not found. Please ensure the model file is in the correct directory.")
    model = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model = None


# Define the image transformations (should match the training script)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define the class labels
CLASSES = [
    "Annual Crop", "Forest", "Herbaceous Vegetation", "Highway",
    "Industrial", "Pasture", "Permanent Crop", "Residential",
    "River", "Sea/Lake"
]

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Handles the image detection request."""
    if not model:
        return jsonify({'error': 'Model is not loaded. Please check server logs.'}), 500

    if 'image' not in request.json:
        return jsonify({'error': 'No image data found in the request.'}), 400

    # Get the base64-encoded image data from the request
    image_data = request.json['image'].split(',')[1]

    try:
        # Decode the base64 string and open it as an image
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess the image
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_label = CLASSES[predicted_idx.item()]

        print(f"Prediction successful: {predicted_label}")
        return jsonify({'prediction': predicted_label})

    except Exception as e:
        print(f"An error occurred during detection: {e}")
        return jsonify({'error': 'An error occurred during processing.'}), 500

if __name__ == '__main__':
    # Runs the Flask app
    # Use 0.0.0.0 to make it accessible on your local network
    app.run(host="127.0.0.1", port=5001, debug=True)

