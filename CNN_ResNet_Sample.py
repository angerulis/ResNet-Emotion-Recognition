# -*- coding: utf-8 -*-
"""
CPSC-4117EL: CNN using Transfer Learning (from ResNet18)

A basic example that outlines how you can use a webcam to collect data, 
train a simple CNN using <transfer learning>, and test the model using 
real-time camera input.

Flow of the program:

1. Collect Data using Webcam:
- press 'c' key to capture images as input, by default 10 images for each classes

2. Train our CNN using Transfer Learning (from ResNet - residual networks):
- we nitialize the ResNet-18 model (18 layers) with weights pre-trained on the ImageNet dataset. 
We freeze all layers (i.e., you don't update their weights during training) 
except for the final fully connected layer, which will adapt to our 
three specific classes. This approach is effective because the model leverages
the general features learned from the ImageNet dataset and only 
fine-tunes the decision-making (final layer) for your specific classes.

https://pytorch.org/vision/main/models/resnet.html

This does not mean we won't be training on our own dataset. 
Instead, we'll be starting with a model that already has learned 
features from ImageNet, which can give us a significant boost, 
especially when our dataset is small. This method is called "Transfer Learning"

https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/

ResNet-18: 1 (initial conv) + 2x2 (first set) + 2x2 (second set) + 
2x2 (third set) + 2x2 (fourth set) + 1 (fully connected) = 18 layers

3. Test the Model with Webcam Input:
- point your webcam to the image to see the classification in real-time.
- press 'q' key to quit
"""

# Import necessary libraries
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, TensorDataset

# Check if 'data' directory exists. If not, create directories for three classes.
if not os.path.exists('data'):
    os.makedirs('data/class1')
    os.makedirs('data/class2')
    os.makedirs('data/class3')

# Function to collect data using the webcam.
def collect_data(class_name, num_samples=10):
    # Initialize the webcam.
    cap = cv2.VideoCapture(0)
    count = 0
    print('Collecting', class_name)
    while True:
        # Read frames from the webcam.
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show the current frame for data collection.
        cv2.imshow('Collect Data', frame)

        # When 'c' key is pressed, save the current frame as an image.
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite(f'data/{class_name}/{class_name}_{count}.jpg', frame)
            print(f'sample {count+1}')
            count += 1
            # Stop collection after reaching the desired number of samples.
            if count >= num_samples:
                break

    # Release the webcam and close OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()

# Collect samples for each class.
collect_data('class1', 10)
collect_data('class2', 10)
collect_data('class3', 10)

# Setup device for PyTorch (use CUDA if available).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained ResNet18 model.
model = resnet18(pretrained = True)
# Freeze all layers in the model. We'll only train the final layer.
for param in model.parameters():
    param.requires_grad = False

# Modify the final fully connected layer to classify 3 classes.
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(device)

# Define the loss function and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Define the transformations for preprocessing the images.
data_transforms = transforms.Compose([
    transforms.ToPILImage(),  # Convert image to PIL Image format.
    transforms.Resize((224, 224)),  # Resize to fit ResNet18's input size.
    # Apply random augmentations to increase dataset variety.
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    # Normalize according to values suitable for ResNet18.
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Lists to hold processed images and their labels.
data = []
labels = []

# Read and preprocess images from each class directory.
for class_name in ['class1', 'class2', 'class3']:
    for filename in os.listdir(f'data/{class_name}'):
        img = cv2.imread(f'data/{class_name}/{filename}')
        img = data_transforms(img)
        data.append(img)
        # Assign a numeric label based on class.
        if class_name == 'class1':
            labels.append(0)
        elif class_name == 'class2':
            labels.append(1)
        else:
            labels.append(2)

# Convert data and labels to PyTorch tensors.
data = torch.stack(data)
labels = torch.tensor(labels)

# Create a PyTorch dataset and data loader.
dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Training loop for the model.
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # Zero out any gradient from the previous step.
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # Compute the gradients.
        optimizer.step()  # Update the weights.

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Use the trained model for live predictions using webcam feed.
cap = cv2.VideoCapture(0)


# Save the model's state dictionary
model_path = 'model_state.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
model.load_state_dict(torch.load(model_path))

model.eval()  # Set the model to evaluation mode if using for inference


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame and get predictions.
    # Transform the image frame: increases the dimensions of the tensor 
    # by adding an extra dimension at the 0th position to create
    # a batch of one image for ResNet18
    input_img = data_transforms(frame).unsqueeze(0).to(device)
    with torch.no_grad(): #  It doesn't need to keep track of gradients.
        output = model(input_img)
        _, pred = torch.max(output, 1) # get the predicted class

    # Map numeric predictions back to class names.
    if pred == 0:
        label = 'class1'
    elif pred == 1:
        label = 'class2'
    else:
        label = 'class3'
    
    # Display the predictions on the frame.
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    cv2.imshow('Predictions', frame)

    # Exit loop on pressing 'q' key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup resources.
cap.release()
cv2.destroyAllWindows()
