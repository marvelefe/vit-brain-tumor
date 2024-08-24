import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
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
    
    return predicted.item(), confidence.item(), probabilities.cpu().numpy(), image

# Path to the image
image_path = './data/Testing/pituitary/Te-pi_0018.jpg'  # replace with your image path

# Predict the class
predicted_class, confidence, probabilities, image = predict_image(image_path, model, data_transforms, device)

# Define class names (assuming you know the class order)
train_dataset = ImageFolder('./data/Training', transform=data_transforms)
class_names = train_dataset.classes

# Print predicted class and confidence
print(f'Predicted class: {class_names[predicted_class]}')
print(f'Confidence: {confidence * 100}%')

# Create subplots: one for the image, one for the bar chart
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Display the image on the left subplot
axs[0].imshow(image)
axs[0].set_title(f'Predicted: {class_names[predicted_class]}')
axs[0].axis('off')

# Display the bar chart on the right subplot
print(probabilities)
multiply = lambda items: list(map(lambda x: x * 100, items))

rects = axs[1].barh(range(len(class_names)), multiply(probabilities),  align='center',
                     height=0.5, color='black')
axs[1].set_yticks(range(len(class_names)))
axs[1].set_yticklabels(class_names)
axs[1].set_xlim([0, 101]) 
axs[1].set_title('Confidence Level')
axs[1].xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)

rect_labels = []
# Lastly, write in the ranking inside each bar to aid in interpretation
for rect in rects:
    # Rectangle widths are already integer-valued but are floating
    # type, so it helps to remove the trailing decimal point and 0 by
    # converting width to int type
    width = int(rect.get_width())

    rankStr = f"{width}%"
    # The bars aren't wide enough to print the ranking inside
    if width < 40:
        # Shift the text to the right side of the right edge
        xloc = 5
        # Black against white background
        clr = 'black'
        align = 'left'
    else:
        # Shift the text to the left side of the right edge
        xloc = -5
        # White on magenta
        clr = 'white'
        align = 'right'

    # Center the text vertically in the bar
    yloc = rect.get_y() + rect.get_height() / 2
    label = axs[1].annotate(rankStr, xy=(width, yloc), xytext=(xloc, 0),
                        textcoords="offset points",
                        ha=align, va='center',
                        color=clr, weight='bold', clip_on=True)
    rect_labels.append(label)

# Save the figure to disk
output_path = './prediction_result.png' 
plt.savefig(output_path, bbox_inches='tight')
print(f'Figure saved to {output_path}')

# Show the figure
plt.show()
