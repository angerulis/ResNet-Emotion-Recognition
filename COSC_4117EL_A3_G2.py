# -*- coding: utf-8 -*-
"""
CPSC-4117EL: CNN using Transfer Learning (from ResNet18)

Objective:
The objective is to develop a system capable of detecting facial expressions, 
specifically categorizing them into three classes: happy, neutral, and surprise.
"""

# Import necessary libraries
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import numpy as np
import random
import json
from collections import defaultdict
import time


NUM_EPOCHS = 20
BATCH_SIZE = 64
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# CLASS_NAMES = ['happy', 'neutral', 'surprise']
# MODEL_TYPE = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
MODEL_TYPE = ['resnet18', 'resnet34', 'resnet50']


# Setting random seed, make the result identical each time 
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


"""
Data Acquisition:
The script leverages the FER-2013 dataset from Kaggle, 
a facial expression recognition dataset with images annotated for various emotional expressions.
Use data from FER-2013 https://www.kaggle.com/datasets/msambare/fer2013
"""
class FER2013Dataset(Dataset):
    def __init__(self, directory, class_names, transform=None):
        self.directory = directory
        self.transform = transform
        self.samples = []

        for class_label, class_name in enumerate(class_names):
            class_dir = os.path.join(directory, class_name)
            images = os.listdir(class_dir)#[:1000]  # Limit each class to first n images
            for image_name in images:
                image_path = os.path.join(class_dir, image_name)
                self.samples.append((image_path, class_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, class_label = self.samples[idx]
        image = cv2.imread(image_path)

        if self.transform:
            image = self.transform(image)

        return image, class_label


"""
Data Preprocessing and Augmentation:
The dataset images undergo preprocessing to fit the input specification of the ResNet18 model, which includes resizing to 224x224 pixels. 
These transformations for training step prepare images for the model training and include data augmentation
techniques such as random horizontal flips, rotations, and affine transformations. 
Such augmentations are designed to improve model robustness by simulating variability found in real-world conditions, 
including changes in camera angles and lighting conditions that the model may encounter during live predictions.

Evaluation transforms are simpler compared to training transforms, 
as they do not include augmentation techniques that introduce random variability. 
This ensures consistency and stability during evaluation and live predictions. 
Grayscale conversion is used to match the near-grayscale nature of the training dataset, 
helping to reduce computational complexity and improve live prediction performance.
"""
def get_train_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),  # Convert the image to PIL Image format if it's not already.
        transforms.Resize((224, 224)),  # Resize the image to match the input size of ResNet.
        transforms.Grayscale(num_output_channels=3),  # Convert the image to grayscale

        # Data Augmentation
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomRotation(10),  # Randomly rotate the image within a range of ±10 degrees
        # Adjust brightness and contrast
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Randomly alter brightness and contrast
        # Affine transformation
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Apply random affine transformations
        # Convert to tensor
        transforms.ToTensor(),
        # Normalize the image with mean and std deviation as per ResNet18 requirements
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_eval_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Convert the image to grayscale
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


"""
Model Configuration:
Transfer learning is employed, utilizing the pre-trained weights from ImageNet to accelerate learning and improve model performance on the target task.

ResNet-18: 1 (initial conv) + 2x2 (first set) + 2x2 (second set) + 2x2 (third set) + 2x2 (fourth set) + 1 (fully connected) = 18 layers
ResNet-34: 1 (initial conv) + 3x2 (first set) + 4x2 (second set) + 6x2 (third set) + 3x2 (fourth set) + 1 (fully connected) = 34 layers
ResNet-50: 1 (initial conv) + 3x3 (first set) + 4x3 (second set) + 6x3 (third set) + 3x3 (fourth set) + 1 (fully connected) = 50 layers
ResNet-101: 1 (initial conv) + 3x4 (first set) + 4x4 (second set)  + 23x4 (third set) + 3x4 (fourth set) + 1 (fully connected) = 101 layers
ResNet-152: 1 (initial conv) + 3x4 (first set) + 8x4 (second set)  + 36x4 (third set) + 3x4 (fourth set) + 1 (fully connected) = 152 layers
"""
class ResNetModel:
    def __init__(self, num_classes, model_type='resnet18'):

        if model_type == 'resnet18':
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_type == 'resnet34':
            self.model = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif model_type == 'resnet50':
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_type == 'resnet101':
            self.model = resnet101(weights=ResNet101_Weights.DEFAULT)
        elif model_type == 'resnet152':
            self.model = resnet152(weights=ResNet152_Weights.DEFAULT)
        else:
            raise ValueError("Unsupported model type. Choose from 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.")

        self.freeze_layers()
        self.update_final_layer(num_classes)
        self.model_type = model_type

    def freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def update_final_layer(self, num_classes):
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def to_device(self, device):
        self.model = self.model.to(device)

    def get_model(self):
        return self.model


"""
Training Technique:
Training involves fine-tuning the pre-trained ResNet18 model on the FER-2013 dataset. 
The script may specify the freezing of earlier layers to preserve learned features from ImageNet while later layers are trained to adapt to the new dataset. 
Details on hyperparameters and loss functions used during training would enrich this section.

During training, this function will print out the loss and accuracy for each epoch and elapsed time to the command line. 
"""
def train_model(model, train_loader, criterion, optimizer, num_epochs, device='cpu'):
    model.to_device(device)
    model.get_model().train()  # Set the model to training mode
    total_steps = len(train_loader) * num_epochs

    train_losses = []
    train_accuracies = []
    start_time = time.time() # Record the start time

    progress_bar = tqdm(total=total_steps, desc=f"{model.model_type} Training", leave=False)
    for epoch in range(num_epochs):
        epoch_loss = 0
        all_labels = []
        all_predictions = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model.get_model()(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            progress_bar.update(1)

        average_epoch_loss = epoch_loss / len(train_loader)
        epoch_accuracy = accuracy_score(all_labels, all_predictions)
        train_losses.append(average_epoch_loss)
        train_accuracies.append(epoch_accuracy)

        progress_bar.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    
    progress_bar.close()

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Training completed in: {elapsed_time:.2f} seconds")

    return model, train_losses, train_accuracies, elapsed_time


"""
Model Evaluation:
Post-training, the model's performance is evaluated on a test set distinct from the training data. 
The script uses precision, recall, and F1 score metrics from a classification report provided by scikit-learn 
    to assess the model's predictive capabilities on the three facial expression classes. 

Outputs the classification report to the command line, as well as auc for each class. 
"""
def evaluate_model(model, test_loader, class_names, device='cpu'):
    model.to_device(device)
    model.get_model().eval()  # Set model to evaluation mode

    total = 0
    correct = 0
    all_labels = []
    all_predictions = []
    all_scores = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model.get_model()(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_scores.extend(outputs.cpu().numpy())

    accuracy = 100 * correct / total
    report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)

    # Binarize the labels for AUC calculation
    binarized_labels = label_binarize(all_labels, classes=range(len(class_names)))
    all_scores = np.vstack(all_scores)

    # Calculate AUC for each class
    aucs = {}
    for i, class_name in enumerate(class_names):
        # Compute the AUC for one class
        class_scores = all_scores[:, i]
        class_binarized_labels = binarized_labels[:, i]
        if len(np.unique(class_binarized_labels)) > 1:  # AUC is only defined when there is more than one class present
            aucs[class_name] = roc_auc_score(class_binarized_labels, class_scores)
        else:
            aucs[class_name] = float('nan')  # AUC is not defined for a single class, set to NaN

    # Print model performance metrics
    print(f'Accuracy of the {model.model_type} model on the test images: {accuracy:.2f}%')
    print(classification_report(all_labels, all_predictions, target_names=class_names))

    # Print AUC for each class
    for class_name, auc in aucs.items():
        print(f"AUC of the {model.model_type} model for {class_name}: {auc:.2f}")

    return accuracy, report, aucs


"""
Real-time Prediction:
The trained model is applied to real-time facial expression detection, utilizing webcam input. 
The real-time component includes a user interface that allows for the live classification of facial expressions. 
The user can interact with this feature by pointing the webcam at a subject and observing the classification in real-time. 
The program can be exited at any time by pressing the 'q' key.
"""
# Real-time Prediction
def live_prediction(model, transform, device='cpu'):
    cap = cv2.VideoCapture(0)  # Open webcam
    model.to_device(device)
    model.get_model().eval()  # Set the model to evaluation mode

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame (resize, transformations, etc.)
        processed_frame = transform(frame).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model.get_model()(processed_frame)
            _, predicted = torch.max(output, 1)
            label = CLASS_NAMES[predicted.item()]

        # Display the predictions on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Predictions', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


"""
The main execution function for training and evaluating the ResNet model on the FER2013 dataset.

This function orchestrates the process of setting up the device, initializing the model,
preparing the dataset loaders, defining the loss function and optimizer, and executing
the training process. After training, the model and training metrics is saved for later use. 

The trained model is then loaded for evaluation on a separate test dataset to assess its
performance through various metrics, these metrics is saved for comparison

And finally, the model is used for real-time facial expression prediction using webcam input.
"""
def main():
    # set the seed
    set_seed(2023)

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define data transformations
    train_transforms = get_train_transforms()
    eval_transforms = get_eval_transforms()

    # Set up training dataset and loader
    train_dataset = FER2013Dataset('archive/train', class_names=CLASS_NAMES, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Set up test dataset and loader
    test_dataset = FER2013Dataset('archive/test', class_names=CLASS_NAMES, transform=eval_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Try multiple resnet model and save the results
    metrics = defaultdict(dict)
    for model_type in MODEL_TYPE:

        # Initialize the model
        model = ResNetModel(num_classes=len(CLASS_NAMES), model_type=model_type)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.get_model().fc.parameters(), lr=0.001)

        # Train the model
        model, train_loss, train_accuracy, train_time = train_model(model, train_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, device=device)

        # train metrics
        metrics[model_type]['train_loss'] = train_loss
        metrics[model_type]['train_accuracy'] = train_accuracy
        metrics[model_type]['train_time'] = train_time

        # Save the trained model
        if not os.path.exists('models'):
            os.makedirs('models')

        torch.save(model.get_model().state_dict(), f"models/model_path_{model_type}_paper.pth")
        print(f"Model saved to models/model_path_{model_type}_paper.pth")

        # Load the model
        model.get_model().load_state_dict(torch.load(f"models/model_path_{model_type}_paper.pth"))

        # Evaluate the model
        accuracy, report, auc = evaluate_model(model, test_loader, class_names=CLASS_NAMES, device=device)

        # test metrics
        metrics[model_type]['test_accuracy'] = accuracy
        metrics[model_type]['test_report'] = report
        metrics[model_type]['auc'] = auc

    # Save the metrics
    if not os.path.exists('metrics'):
        os.makedirs('metrics')

    with open('metrics/metrics_paper.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to metrics/metrics_paper.json")

    # # Real-time prediction using webcam
    # # Choose the selected model
    # model_type = 'resnet34'
    # model = ResNetModel(num_classes=len(CLASS_NAMES), model_type=model_type)
    # model.get_model().load_state_dict(torch.load(f"models/model_path_{model_type}_paper.pth"))

    # live_prediction(model, transform=eval_transforms, device=device)


if __name__ == "__main__":
    main()
