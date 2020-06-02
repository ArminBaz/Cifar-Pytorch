import torch
import torchvision
from torchvision import transforms, datasets
from barbar import Bar
import os
from models import *
import torch.nn as nn
import torch.optim as optim

# TODO: Use visdom or tensorboard to visualize training progress

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
trainset = torchvision.datasets.CIFAR10(root="../data/train", train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root="../data/test", train=False, download=True, transform=transform_test)


'''
DataLoader: Data Loader goes along with transforms as one of the cooler things that pytorch brings to the table. This extends the Dataset object by acting as a "feeder" when you want to train your model.

Again this is nice because it allows us to define a batch size, shuffle the data, and more.

At the very least it's just a nice way to feed the data
'''
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

print('Finished preparing data')

# TODO: Add Argument Parsing functionality (learning rate, epochs, checkpoints, model_type)
num_epochs = 10

# GPU support
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Fetching model...')

net = BaseNet()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
m
# TODO: Actually save the best model in checkpoints during testing
# Helper function to save best model
def save_model(state, model_name):
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    torch.save(state, './checkpoint/' + model_name)


# Train the model
for epoch in range(num_epochs):
    print('Epoch: {}'.format(epoch+1))
    running_loss = 0.0

    for batch_id, (inputs, labels) in enumerate(Bar(train_loader)):
        # zero the paramater gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Normalizing the loss by the total number of train batches
    running_loss /= len(train_loader)

    # Calculate training set accuracy
    train_accuracy = calculate_accuracy(train_loader)

    print("Epoch: {0} | Loss: {1} | Training Accuracy: {2}".format(epoch+1, running_loss, train_accuracy))


# test the model
print("Testing the model...")
global best_acc
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_id, (inputs, targets) in enumerate(Bar(test_loader)):
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
acc = 100.*correct/total
print('The test accuracy of the model is: {}'.format(acc))
