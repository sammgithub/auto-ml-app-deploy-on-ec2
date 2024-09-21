import os
import time  # Import time module
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.nn.functional as F

# Initialize the process group
def init_process_group():
    dist.init_process_group(backend='nccl')

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # Initialize distributed training
    init_process_group()
    device = torch.device(f'cuda:{dist.get_rank()}')
    
    # Create the model
    model = SimpleCNN().to(device)
    model = nn.parallel.DistributedDataParallel(model)

    # Load the dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, sampler=train_sampler)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    start_time = time.time()  # Record the start time
    for epoch in range(5):  # Number of epochs
        train_sampler.set_epoch(epoch)  # Shuffle data each epoch
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/5], Loss: {running_loss / len(train_loader):.4f}')
    
    total_time = time.time() - start_time  # Calculate total training time
    print(f'Total training time: {total_time:.2f} seconds')  # Print total training time

if __name__ == "__main__":
    main()
