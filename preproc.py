import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

import numpy as np
from matplotlib import pyplot as plt



def dsm_score_estimation(scorenet, samples, sigma=1.):
    perturbed_samples = samples + torch.randn_like(samples) * sigma
    target = - 1 / (sigma ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)

    return loss

def load_and_preprocess_data(dataset_name, batch_size, augment=False):
    """Load, preprocess and return data for training and testing.
    
    Args:
        dataset_name: str, name of dataset from {'mnist', 'cifar10'}.
        batch_size: int, batch size used for training and testing.
        augment: bool, if True, apply data augmentation defined by the augmentation transforms.
    Returns:
        Training and test dataloader objects.
    """

    # Define transformations that will be appied to images
    train_transforms = transforms.Compose([transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.ToTensor()])

    if augment is True:
        augmentation_transforms = [transforms.RandomCrop(28, padding=4),
                                   transforms.RandomHorizontalFlip(), 
                                   transforms.RandomGrayscale(),
                                   transforms.GaussianBlur((3,3)),
                                   transforms.RandomRotation(np.random.uniform(low=-3.14, high=3.14))]
        train_transforms = transforms.Compose(augmentation_transforms + [transforms.ToTensor()])
        
    # Load train and test datasets
    if dataset_name == 'mnist':
        train_dataset = torchvision.datasets.MNIST(root='./data', 
                                                   train=True,
                                                   download=True, 
                                                   transform=train_transforms)
        test_dataset = torchvision.datasets.MNIST(root='./data', 
                                                  train=False,
                                                  download=True, 
                                                  transform=test_transforms)
    elif dataset_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                                     train=True,
                                                     download=True, 
                                                     transform=train_transforms)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                                    train=False,
                                                    download=True, 
                                                    transform=test_transforms)
    else:
        raise Exception("The dataset name must be element of {'mnist', 'cifar10'}.")

    # Create dataset loaders
    trainset_loader = torch.utils.data.DataLoader(train_dataset, 
                                                  batch_size=batch_size,
                                                  shuffle=True, 
                                                  num_workers=2)
    testset_loader = torch.utils.data.DataLoader(test_dataset, 
                                                 batch_size=batch_size,
                                                 shuffle=False, 
                                                 num_workers=2)

    # Print data info
    print(f"Dataset: {dataset_name} \n", 
          f"Image shape: {train_dataset[0][0].numpy().shape} \n",
          f"Number of train images: {len(train_dataset)} \n",
          f"Number of test images: {len(test_dataset)} \n",
          f"Number of classes: {len(np.unique(train_dataset.targets))} \n")
    
    # Visualize a batch of input examples 
    images, labels = iter(trainset_loader).next()
    image_grid = torchvision.utils.make_grid(images)
    plt.imshow(np.transpose(image_grid, (1, 2, 0)))
    plt.axis('off')
    plt.show()

    return (trainset_loader, testset_loader)



def training_and_eval(dataset_name, model, optimizer, batch_size, num_epochs, 
                        augment=False, need_summary = False, sigma = None):
    """Training and testing.
    
    Args:
        dataset_name: str, name of dataset from {'mnist', 'cifar10'}.
        model: instance of MLPModel or CNNModel class.
        optimizer: instance of any optimizer defined in the torch.nn.optim module.
        batch_size: int, batch size used for training and testing.
        num_epochs: int, number of training epochs.

        augment: bool, if True, apply augmentation transforms in load_and_preprocess_data function.
    Returns:
        model: a trained model that is instance of MLPModel or CNNModel class.
        history: dict that contains the loss and accuracy history.
    """

    # Get the train and test data
    train_loader, test_loader = load_and_preprocess_data(dataset_name, 
                                                         batch_size, 
                                                         augment=augment)

    # Train the model
    history = {'train_loss': [],
               'train_accuracy': [],
               'test_loss': [],
               'test_accuracy': []}

    print(f"Model summary")
    if need_summary:
    	summary(model, input_size=model.input_size)
    print(f"Train the model on {dataset_name} dataset for {num_epochs} epochs...\n")


    length = 10
    ratio = pow(0.01/10, 1/9)
    start = 10
    progression = np.array([start * ratio**i for i in range(length)])
    epses = np.logspace(-1, -5, 9)
    sigma = 0.1 if sigma is None else sigma

    device = torch.cuda.current_device()
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        model = model.train()

        # Train
        for i, (images, labels) in enumerate(train_loader):
            #images_o = images.to(device)
            #eps_ = (eps**2/progression[-1]**2)*2e-5
            #images = images_o + torch.normal(0, std=sigma,size=images.shape).to(device)
            #labels = (images-images_o)/sigma
            #for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = images.to(device)
            


            # Forward pass + backprop + loss calculation
            loss = dsm_score_estimation(model, images, sigma = sigma)
            optimizer.zero_grad()
            loss.backward()

            # Update model params
            optimizer.step()

            train_loss += loss.detach().item()
            #train_acc += get_accuracy(predictions, labels, batch_size)

        train_loss = train_loss / (i+1)
        train_acc = train_acc / (i+1)
        print(f"Epoch: {epoch, sigma} | Train loss: {train_loss} | Train accuracy: {train_acc}")  

        model.eval()
        # Evaluate on test set
        test_loss = 0.0
        test_acc = 0.0
        for i, (images, labels) in enumerate(test_loader):
            #images_o = images.to(device)
            #eps_ = (eps**2/progression[-1]**2)*2e-5
            #images = images_o + torch.normal(0, std=sigma,size=images.shape).to(device)
            #labels = (images-images_o)/sigma
            images = images.to(device)

            loss = dsm_score_estimation(model, images)
            test_loss += loss.detach().item()
            #test_acc += get_accuracy(predictions, labels, batch_size)
        test_loss = test_loss / (i+1)
        test_acc = test_acc / (i+1)
        print(f" \t  Test loss: {test_loss} | Test accuracy: {test_acc}")
        model.train() 

        # Add results to the history dict
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_acc)
   
    return model, history