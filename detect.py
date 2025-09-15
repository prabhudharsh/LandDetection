import torch
import torch.nn as nn

# Define SimpleCNN class exactly like in training
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 🔹 Load the full model (allow pickled objects)
model = torch.load("eurosat_model.pkl", weights_only=False)
model.eval()

print("Model loaded successfully!")

from torchvision import transforms
from PIL import Image
import torch

# Transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# Replace 'test.jpg' with your image path
img = Image.open("test.png").convert("RGB")

# Apply transforms and add batch dimension
img_tensor = transform(img).unsqueeze(0)  # shape: [1, 3, 64, 64]


model.eval()  # make sure model is in evaluation mode
with torch.no_grad():
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)

classes = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

print("Predicted class index:", predicted.item())
print("Predicted label:", classes[predicted.item()])
