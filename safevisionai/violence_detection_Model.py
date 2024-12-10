import torch
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch import nn, optim
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
import json

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

# Logging and Plotting Setup
train_losses, val_losses = [], []
log_file = "training_log.json"
log_data = {"epochs": []}

# Training
epochs = 10
print("Starting training...")
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    model.train()
    train_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss_avg = train_loss / len(train_loader)
    train_losses.append(train_loss_avg)
    print(f"Epoch {epoch+1}: Train Loss: {train_loss_avg:.4f}")

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

    val_loss_avg = val_loss / len(val_loader)
    val_losses.append(val_loss_avg)
    print(f"Epoch {epoch+1}: Val Loss: {val_loss_avg:.4f}")
    print("Validation Classification Report:")
    print(classification_report(y_true, y_pred, target_names=train_dataset.classes))

    # Save Best Model and Early Stopping
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        no_improve_epochs = 0
        best_model_path = f"best_violence_detection_model_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), best_model_path)
        print(f"Validation loss improved. Model saved to {best_model_path}.")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("Early stopping triggered. Stopping training.")
            break

    # Log Epoch Information
    log_data["epochs"].append({
        "epoch": epoch + 1,
        "train_loss": train_loss_avg,
        "val_loss": val_loss_avg
    })

# Save Final Model
final_model_path = "final_violence_detection_model.pth"
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}.")

# Save Logs to File
with open(log_file, "w") as f:
    json.dump(log_data, f)
print(f"Training log saved to {log_file}.")

# Plot Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid()
plt.savefig("loss_plot.png")
plt.show()

# Summary Information
print("\nTraining Summary:")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Best Model Path: {best_model_path}")
print(f"Final Model Path: {final_model_path}")
print(f"Training Log Path: {log_file}")
print("Loss Plot saved to loss_plot.png")
