## Cifar10 with Pytorch

### Introduction
The main objective is to get practice with Pytorch. To do this I am creating different well known models and applying each one to the Cifar10 dataset. <br>

Because I don't have access to a GPU, I am training these models using Google Colab (Thank you Google) for 100 epochs <br>

Also, because I don't feel like training models all day. I decided that I would cut the training once the model achieves ~90% training accuracy. Obviously, this is not the best way to compare models, but I think it is still interesting nonetheless and seeing the differences in training speed should be enough of an indicator at how well the model is able to generalize complex data.

### Requirements
* python 3.7
* torchvision 0.5.0
* numpy 1.17.2
* matplotlib 3.1.1
* barbar 0.2.1

### To Do:
- [x] BaseNet
- [x] ResNet
- [ ] DenseNet
- [ ] Inception V3
- [ ] VGG
- [ ] etc...

### Accuracies
- BaseNet ~ 90% training accuracy reached at epoch 88
- ResNet18 ~ 90% training accuracy reached at epoch 36
