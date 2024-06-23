import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from vit_pytorch import ViT
from torchvision.datasets import ImageFolder
import torch.nn as nn
import matplotlib.pyplot as plt
from transformer import TumorClassifierViT

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = TumorClassifierViT(num_classes=4)
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)
model.eval()

# Define the transformation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict a single image
def predict_image(image_path, model, transform, device):
    # Open image
    image = Image.open(image_path).convert('RGB')
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)
    # Move to device
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
    
    return predicted.item(), confidence.item(), image

# Path to the image
image_path = './data/Testing/pituitary/Te-pi_0011.jpg'  # replace with your image path

# Predict the class
predicted_class, confidence, image = predict_image(image_path, model, data_transforms, device)

# Define class names (assuming you know the class order)
train_dataset = ImageFolder('./data/Training', transform=data_transforms)
class_names = train_dataset.classes  # or manually define: ['Class0', 'Class1', 'Class2', 'Class3']

# Print predicted class and confidence
print(f'Predicted class: {class_names[predicted_class]}')
print(f'Confidence: {confidence:.2f}')

# Plot the image with predicted label and confidence
plt.imshow(image)
plt.title(f'Predicted: {class_names[predicted_class]} ({confidence:.2f})')
plt.axis('off')

# Save the figure to disk
output_path = './prediction_result.png'  # specify the desired output path
plt.savefig(output_path, bbox_inches='tight')
print(f'Figure saved to {output_path}')
