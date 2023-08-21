import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets, models

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the transformation for preprocessing the data
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data='dataset/training_set/'
test_data='dataset/test_set/'

# Set the batch size and number of epochs
batch_size = 32
num_epochs = 20

# Load the dataset
train_dataset = datasets.ImageFolder(train_data, transform=train_transform)
test_dataset = datasets.ImageFolder(test_data, transform=test_transform)

# Compute class weights to address class imbalance
class_counts = torch.bincount(torch.tensor(train_dataset.targets))
total_samples = len(train_dataset)
class_weights = 1.0 / class_counts

# Create data loaders with weighted sampling
weights = class_weights[train_dataset.targets]
print("Class weights: ", weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)

# Modify the last fully connected layer to match the number of classes
num_classes = 4
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Set the model to training mode
model.train()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print statistics
        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            train_accuracy = 100 * correct / total
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                  f'Loss: {running_loss / 10:.4f}, Train Accuracy: {train_accuracy:.2f}%')
            running_loss = 0.0

print('Training finished.')

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')
