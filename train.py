import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
import os
import json
import time
from typing import Dict, Any

def train_model(
    progress: Dict[str, Any],
    data_dir: str = "datasets",
    models_dir: str = "models",
    num_epochs: int = 10,
    learning_rate: float = 0.001,
):
    """
    Trains an image classification model.

    Args:
        progress (Dict[str, Any]): Object for reporting training progress.
        data_dir (str): Path to the dataset directory.
        models_dir (str): Directory to save trained models.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
    
    Returns:
        str: Path to the directory where the trained model and metadata are saved.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    full_dataset = ImageFolder(root=data_dir, transform=transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    if num_classes < 2:
        raise ValueError("Dataset must contain at least two classes for training.")

    # Split dataset into training and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Load a pre-trained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Replace the classifier for transfer learning
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize progress
    progress["status"] = "Training"
    progress["total_epochs"] = num_epochs
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = corrects.double() / len(val_loader.dataset)

        log_message = (
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f} | "
            f"Val Acc: {val_epoch_acc:.4f}"
        )
        print(log_message)

        # Update progress
        progress["current_epoch"] = epoch + 1
        progress["log"] = log_message

    # Save the model
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_save_dir = os.path.join(models_dir, timestamp)
    os.makedirs(model_save_dir, exist_ok=True)

    model_path = os.path.join(model_save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)

    # Save class names mapping
    class_map_path = os.path.join(model_save_dir, "class_map.json")
    class_to_idx = full_dataset.class_to_idx
    with open(class_map_path, 'w') as f:
        json.dump(class_to_idx, f, indent=4)
        
    final_message = f"Training complete. Model saved to {model_save_dir}"
    print(final_message)
    
    # Final progress update
    progress["status"] = "Completed"
    progress["log"] = final_message
    
    return model_save_dir 