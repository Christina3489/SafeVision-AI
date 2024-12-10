import torch
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch import nn, optim
from sklearn.metrics import classification_report
import os

# Paths
train_dir = "C:/Users/chris/Downloads/violence_dataset/processed_data/train"
val_dir = "C:/Users/chris/Downloads/violence_dataset/processed_data/val"

# Verify Dataset Paths
assert os.path.exists(train_dir), f"Training directory not found: {train_dir}"
assert os.path.exists(val_dir), f"Validation directory not found: {val_dir}"

# Data Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Datasets
print("Loading training dataset...")
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
print(f"Training dataset loaded with {len(train_dataset)} samples.")

print("Loading validation dataset...")
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
print(f"Validation dataset loaded with {len(val_dataset)} samples.")

# Data Loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

# Check Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Using CPU.")

# Load Pretrained Model (ResNet18)
print("Initializing model...")
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # Binary classification: Violence, Non-Violence
model.to(device)
print("Model initialized successfully.")

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early Stopping Variables
best_val_loss = float('inf')
patience = 3
no_improve_epochs = 0

# Training
epochs = 10
print("Starting training...")
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    model.train()
    train_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Training batch {batch_idx+1}/{len(train_loader)}...", end='\r')
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate Metrics
    val_loss_avg = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}: Val Loss: {val_loss_avg:.4f}")
    print("Validation Classification Report:")
    print(classification_report(y_true, y_pred, target_names=train_dataset.classes))

    # Save Best Model and Early Stopping
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        no_improve_epochs = 0
        torch.save(model.state_dict(), "best_violence_detection_model.pth")
        print("Validation loss improved. Model saved.")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("Early stopping triggered. Stopping training.")
            break

# Save Final Model
torch.save(model.state_dict(), "final_violence_detection_model.pth")
print("Training complete.")
