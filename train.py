# Import libararies
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
import all_functions

parser = argparse.ArgumentParser(description='Train.py')


parser.add_argument('data_dir', action="store", default="./flowers/")
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=3)
parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)



args = parser.parse_args()

root = args.data_dir
path = args.save_dir
lr = args.learning_rate
arch = args.arch
dropout = args.dropout
hidden_layer1 = args.hidden_units
device = args.gpu
epochs = args.epochs

def main():
    trainloader, valloader, testloader = all_functions.load_data(root)
    model, optimizer, criterion = all_functions.build_network(arch,dropout,hidden_layer1,lr,device)
    all_functions.train_model(model, optimizer, criterion, epochs, 40, trainloader, device)
    all_functions.save_checkpoint(model,path,arch,hidden_layer1,dropout,lr)
    print("Finished! :)")


if __name__== "__main__":
    main()
