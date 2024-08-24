import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
from vit_pytorch import ViT
from vit_pytorch.vit import Transformer
from tqdm import tqdm
from transformer import TumorClassifierViT
from sklearn.metrics import confusion_matrix
import seaborn as sns
from prettytable import PrettyTable
import arrow
  
tableData = PrettyTable(['Epoch', 'Time Elapsed (HH:mm:ss)', 'Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy'])

# Set the device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data preprocessing and augmentation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder('./data/Training', transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = ImageFolder('./data/Testing', transform=data_transforms)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load a batch of images and labels for visualization
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Convert images to numpy arrays and denormalize
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
images = (images.numpy().transpose((0, 2, 3, 1)) * std + mean).clip(0, 1)

# Create a grid of images
num_images = len(images)
rows = int(np.ceil(num_images / 4))
fig, axes = plt.subplots(rows, 4, figsize=(15, 15))

# Plot images with labels
for i, ax in enumerate(axes.flat):
    if i < num_images:
        ax.imshow(images[i])
        ax.set_title(f'Label: {train_dataset.classes[labels[i]]}')
    ax.axis('off')
 
plt.savefig('classes.png')  
plt.close()



model = TumorClassifierViT(num_classes=4).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Initialize lists to store training history
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
num_epochs = 100
best_val_accuracy = 0.0

startTime = arrow.now().timestamp()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []


    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    val_loss /= len(val_loader)
    val_accuracy = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], ' f'Time elapsed: {arrow.get(arrow.now().timestamp() - startTime).format("HH:mm:ss")}'
           f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2%}, '
           f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2%}')

    #  'Epoch', 'Time Elapsed', 'Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy'
    currentEpoch = epoch + 1
    if currentEpoch % 5 == 0 or currentEpoch == 1:
        tableData.add_row([f'{currentEpoch}', f'{arrow.get(arrow.now().timestamp() - startTime).format("HH:mm:ss")}', f'{train_loss:.4f}', f'{train_accuracy:.4f}', f'{val_loss:.4f}' , f'{val_accuracy:.2%}'])

    # Save the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')

accuracy = correct / total
print(f'Validation Accuracy: {accuracy:.2%}')

print(tableData)

# Visualize training history
plt.figure(figsize=(10, 7))
plt.subplot()
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss History')
plt.tight_layout()
plt.savefig('loss.png')  
plt.close()

plt.figure(figsize=(10, 7))
plt.subplot()
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy History')
plt.tight_layout()
plt.savefig('accuracy.png')  
plt.close()

# Visualize confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=val_dataset.classes, yticklabels=val_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  
plt.close()



