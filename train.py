import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import random

class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

MODEL_PATH = "mnist_model.pt"

def generate_handwritten_sample(digit):
    img = Image.new('L', (100, 100), color=0)
    draw = ImageDraw.Draw(img)
    stroke = random.randint(12, 18)
    
    if digit == 0:
        draw.ellipse([20, 15, 80, 85], outline=255, width=stroke)
    elif digit == 1:
        draw.line([(50, 15), (50, 85)], fill=255, width=stroke)
        if random.random() > 0.5:
            draw.line([(35, 30), (50, 15)], fill=255, width=stroke)
    elif digit == 2:
        # Fixed curves for 2 so it doesn't close like an 8
        draw.arc([20, 15, 80, 55], 180, 0, fill=255, width=stroke)
        draw.line([(80, 35), (20, 85)], fill=255, width=stroke)
        draw.line([(20, 85), (85, 85)], fill=255, width=stroke)
    elif digit == 3:
        draw.arc([20, 15, 80, 50], 270, 90, fill=255, width=stroke)
        draw.arc([20, 45, 80, 85], 270, 90, fill=255, width=stroke)
    elif digit == 4:
        draw.line([(25, 15), (25, 55)], fill=255, width=stroke)
        draw.line([(25, 55), (80, 55)], fill=255, width=stroke)
        draw.line([(65, 15), (65, 85)], fill=255, width=stroke)
    elif digit == 5:
        draw.line([(80, 15), (25, 15)], fill=255, width=stroke)
        draw.line([(25, 15), (25, 45)], fill=255, width=stroke)
        draw.arc([25, 40, 80, 85], 270, 90, fill=255, width=stroke)
    elif digit == 6:
        draw.ellipse([20, 45, 80, 85], outline=255, width=stroke)
        draw.arc([20, 15, 80, 60], 90, 200, fill=255, width=stroke)
    elif digit == 7:
        draw.line([(20, 15), (80, 15)], fill=255, width=stroke)
        draw.line([(80, 15), (35, 85)], fill=255, width=stroke)
    elif digit == 8:
        # Clearly separate top and bottom loops for 8
        draw.ellipse([28, 15, 72, 48], outline=255, width=stroke)
        draw.ellipse([22, 45, 78, 85], outline=255, width=stroke)
    elif digit == 9:
        draw.ellipse([20, 15, 80, 55], outline=255, width=stroke)
        draw.line([(80, 35), (80, 85)], fill=255, width=stroke)

    # Blur slightly to mimic soft mouse/brush strokes
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.1307) / 0.3081
    return arr

print("Generating offline handwritten data (0 bytes downloaded)...")
images_list, labels_list = [], []

for digit in range(10):
    for _ in range(800):
        images_list.append(generate_handwritten_sample(digit))
        labels_list.append(digit)

x_tensor = torch.tensor(np.array(images_list)).unsqueeze(1)
y_tensor = torch.tensor(np.array(labels_list), dtype=torch.long)

dataset = TensorDataset(x_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = DigitCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training model offline...")
for epoch in range(6):
    for imgs, lbls in loader:
        optimizer.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), MODEL_PATH)
print("Done! Restart app.py now.")