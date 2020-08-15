import torch
import torchvision
from torchvision import transforms, datasets
from barbar import Bar
import os
import argparse
from models import *
import torch.nn as nn
import torch.optim as optim

# TODO: Add more arguments
##   Argument parsing   ##
# create the parser
parser = argparse.ArgumentParser(description='Arguments for training (model and epoch)')

# add the arguments
parser.add_argument('--model', default=BaseNet, 
    help='Model to train/test on (BaseNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152', 
    action='store')
parser.add_argument('--num_epoch', default=100, help='Number of epochs for model to train for', action='store')
parser.add_argument('--download_data', default=True, help='Download data? (Ture/False)', action='store')

# execute the parse_args() method
args = parser.parse_args()

# TODO: Use visdom or tensorboard to visualize training progress

# Function to get data if not already downloaded
def get_data(data_bool):
    print('Preparing data...')
    '''
    Transforms: Pytorch transforms provide an easy way for us (the user) to manipulate the data to our liking.
    This is beautiful for pre-processing because it saves us time
    '''
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    '''
    Datasets: Torchvision comes pre-loaded with many common computer vision datasets, including the benchmark cifar10 dataset.

    The Dataset class also allows you to easily create your own dataset wrappers with the pytorch API,
    all you have to do is override the two subclass functions {len and getitem}
    '''
    trainset = torchvision.datasets.CIFAR10(root="../data/train", train=True, download=data_bool, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root="../data/test", train=False, download=data_bool, transform=transform_test)


    '''
    DataLoader: Data Loader goes along with transforms as one of the cooler things that pytorch brings to the table. This extends the Dataset object by acting as a "feeder" when you want to train your model.

    Again this is nice because it allows us to define a batch size, shuffle the data, and more.

    At the very least it's just a nice way to feed the data
    '''
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    return trainloader, testloader

    print('Finished preparing data')

##   Use the arguments to get data and define variables   ##
num_epochs = args.num_epoch
train_loader, test_loader = get_data(args.download_data)

# model selection
# TODO: Find a better way to do this
if args.model = 'BaseNet':
    net = BaseNet()
elif args.model = 'ResNet18':
    net = ResNet18()
elif args.model = 'ResNet34':
    net = ResNet34()
elif args.model = 'ResNet50':
    net = ResNet50()
elif args.model = 'ResNet101':
    net = ResNet101()
elif args.model = 'ResNet152':
    net = ResNet152()

# GPU support
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Fetching model...')

net = BaseNet()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# TODO: Actually save the best model in checkpoints during testing
# Helper function to save best model
def save_model(state, model_name):
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    torch.save(state, './checkpoint/' + model_name)

# Train the model
def train_model(net):
    total = 0
    correct = 0

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch+1}')
        running_loss = 0.0

        for batch_id, (inputs, labels) in enumerate(Bar(train_loader)):
            # Send input tensors to device (needed for GPU support)
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the paramater gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Normalizing the loss by the total number of train batches
        running_loss /= len(train_loader)

        # Calculate training set Accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        train_accuracy = 100.*correct/total

        print(f"Epoch: {epoch+1} | Loss: {running_loss} | Training Accuracy: {train_accuracy}")


# test the model
def test_model(net):    
    print("Testing the model...")
    global best_acc
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_id, (inputs, targets) in enumerate(Bar(test_loader)):
            # Send input tensors to device (needed for GPU support)
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    acc = 100.*correct/total
    print(f'The test accuracy of the model is: {acc}')

# Train and test the model
train_model()
test_model()