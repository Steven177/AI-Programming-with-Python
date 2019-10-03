import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
from collections import OrderedDict
import argparse

arch = {"vgg16":25088,
        "densenet121":1024, 
        "alexnet":9216
        }

def transform_image(root):

    data_dir = root
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms ={
    'train_transforms':transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
    'test_transforms' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])]),
    'validation_transforms' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])}
    image_datasets = {'train_data' : datasets.ImageFolder(data_dir + '/train', transform=data_transforms['train_transforms']),
    'test_data' : datasets.ImageFolder(data_dir + '/test', transform=data_transforms['test_transforms']),
    'valid_data' : datasets.ImageFolder(data_dir + '/valid', transform=data_transforms['validation_transforms'])
                 }
    
    return image_datasets['train_data'] , image_datasets['valid_data'], image_datasets['test_data']



def load_data(root):
    
    data_dir = root    
    train_data,val_data,test_data=transform_image(data_dir)
    
    dataloaders = {'trainloader' : torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True),
    'testloader' : torch.utils.data.DataLoader(test_data, batch_size=32),
    'validloader' : torch.utils.data.DataLoader(val_data, batch_size=32)}
    
    return dataloaders['trainloader'] , dataloaders['validloader'], dataloaders['testloader']


train_data,valid_data,test_data = transform_image('./flowers/')
trainloader,valloader,testloader = load_data('./flowers/')

def build_network(structure='vgg16',dropout=0.5, hidden_layer1 = 4096,lr = 0.001,device='gpu'):

    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("vgg16 or densenet121 or alexnet")


    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(arch[structure],hidden_layer1)),
                          ('relu1', nn.ReLU()),
                          ('d_out1',nn.Dropout(dropout)),
                          ('fc2', nn.Linear(hidden_layer1, 1024)),
                          ('relu2', nn.ReLU()),
                          ('d_out2',nn.Dropout(dropout)),
                          ('fc3', nn.Linear(1024, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))


    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr )

    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()

    return model, criterion, optimizer


def train_model(model, criterion, optimizer, epochs = 3, print_every=60, trainloader=0, device='gpu'):
    # TRAINING
    # Parameters
    epochs = 10
    steps = 0
    print_every = 64

    for e in range(epochs):
        train_loss = 0
        for images, labels in trainloader:
            steps += 1
            if torch.cuda.is_available() and device =='gpu':
                images, labels = images.to('cuda'), labels.to('cuda')

            # Delete all gradients
            optimizer.zero_grad()
            y_hat = model.forward(images)
            loss = criterion(y_hat, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if steps % print_every == 0:
                # VALIDATION
                # val_loss, val_accuracy = validate(model, criterion, valloader, device)

                model.eval()      
                # Parameters
                val_loss = 0
                val_accuracy = 0
                with torch.no_grad():
                    for images, labels in valloader:
                        if torch.cuda.is_available():
                            images, labels = images.to('cuda') , labels.to('cuda')
                            model.to('cuda')

                        # Val Loss
                        out = model.forward(images)
                        loss_val = criterion(out, labels)

                        val_loss += loss_val.item()

                        # Val Accuracy
                        prob = torch.exp(out)
                        top_prob, top_class = prob.topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print("Epoch {}/{}.. ".format(e+1, epochs),
                  "Train loss: {:.3f}.. ".format(train_loss/print_every),
                  "Val loss: {:.3f}.. ".format(val_loss/len(valloader)),
                  "Val accuracy: {:.3f}%".format(val_accuracy/len(valloader) * 100))

                train_loss = 0
                model.train()


def save_checkpoint(model=0,path='checkpoint.pth',arch ='vgg16', hidden_layer1 = 4096,dropout=0.5,lr=0.001,epochs=3):

    model.class_to_idx =  train_data.class_to_idx
    model.cpu
    torch.save({'arch' :arch,
                'hidden_layer1':hidden_layer1,
                'dropout':dropout,
                'lr':lr,
                'epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)


def load_checkpoint(path='checkpoint.pth'):
    checkpoint = torch.load(path)
    lr=checkpoint['lr']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    arch = checkpoint['arch']

    model,_,_ = build_network(arch , dropout,hidden_layer1,lr)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def process_image(image_path='/home/workspace/aipnd-project/flowers/test/1/image_06752.jpg'):
    
    proc_img = Image.open(image_path)

    prepoceess_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    image = prepoceess_img(proc_img)
    return image


def predict(image='/home/workspace/aipnd-project/flowers/test/1/image_06752.jpg', model=0, topk=5,device='gpu'):

    if torch.cuda.is_available() and device =='gpu':
        model.to('cuda')

    img_torch = process_image(image)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if device == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)

    probability = F.softmax(output.data,dim=1)

    return probability.topk(topk)
