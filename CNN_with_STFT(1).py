#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
# from torchvision import datasets
# import torchvision
# import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy.random
from numpy.random import randint
import time
# from ray import tune
# from ray.tune import JupyterNotebookReporter
# from ray.tune.schedulers import ASHAScheduler


# In[2]:


root = 'C:/Users/Joseph/Downloads/'
X_stft_train = np.load(root+'X_stft_train_nfft=1024.npy')
# n_mfcc, ceps_length, n = X_train_mfcc.shape
y_train = np.load(root+'train_labels.npy').squeeze()
# X_test_mfcc = np.load(root+'test_mfcc_nmfcc=5_nfft=8192_hl=512_3obsPerWav.npy')
# n_test = X_test_mfcc.shape[2]


# In[3]:


class stftDataset(Dataset):

    def __init__(self, root, root_dir, transform=None):
        """
        Args:
            root (string): Path to the label array.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = np.load(root+'train_labels.npy')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

#         mfcc_name = os.path.join(self.root_dir, str(idx+1)+'.npy')
#         #reformat mfcc matrix to be closer to square:
#         mfcc = np.load(mfcc_name).reshape((28,35))
#         mfcc = np.expand_dims(mfcc,axis=2)
#         genre = np.floor(np.floor((float(idx))/3)/70)
#         genre = np.array([genre]).astype('float')
        stft = X_stft_train[idx]
        stft = np.expand_dims(stft, axis=2)
        label = np.array([y_train[idx]]).astype('float')
        sample = {'stft': stft, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        stft, label = sample['stft'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        stft = stft.transpose((2, 0, 1))
        return {'stft': torch.from_numpy(stft),
                'label': torch.from_numpy(label)}

stft_dataset = stftDataset(root=root, root_dir=root, transform = ToTensor())
test_prop = 0.1
train_prop = 1-test_prop
n = len(stft_dataset)
trainset, testset = random_split(stft_dataset, [int(n*train_prop), n-int(n*train_prop)], generator=torch.Generator().manual_seed(0))


# In[4]:


trainloader = DataLoader(trainset, batch_size=50, shuffle=True, num_workers=0)


# In[5]:


print(X_stft_train.shape)


# In[6]:


x = torch.rand(5, 3)
print(x)
torch.cuda.is_available()


# In[12]:


def train(net, n_epochs, trainloader, use_gpu, step_size, momentum, show_loss_every):
    if use_gpu:
        net = net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=step_size, momentum=momentum)
#     optimizer = optim.Adam(net.parameters(), lr=step_size)

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a dictionary of {'stft': stft, 'label': label}
            inputs = data['stft']
            labels = data['label'].squeeze()
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float())
            loss = criterion(outputs, labels.long())
            loss.backward()  #Actually calculate gradient
            optimizer.step() #Update the weights

            # print statistics
            running_loss += loss.item()
            if (i+1) % show_loss_every == 0:    # print every show_loss_every mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / show_loss_every))
                running_loss = 0.0
                
    return net


# In[13]:


#Calculate test error
def test(net, testloader, use_gpu):
    net.eval()

    correct = 0
    total = 0

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            #Load data
            inputs = data['stft']
            labels = data['label'].squeeze()
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # calculate outputs by running images through the network
            outputs = net(inputs.float())
            # print(labels)
            loss = nn.CrossEntropyLoss()
            loss(outputs, labels.long())

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(prediction)
    print('Accuracy: %d %%' % (
        100 * correct / total))
    # return outputs.data


# In[14]:


# prediction = test(net, testloader, use_gpu)


# In[15]:


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

n_hidden_layers = 3

class CNN(nn.Module):
    '''
    Convolutional Neural Network
    '''
    #hyperparams: activation, optimization algorithm, max_channels, momentum, step_size, stride
    def __init__(self, n_hidden_layers=n_hidden_layers, activation=nn.ReLU(), stride=1, dim=(3,3), 
                 pool_stride=(1,1), pool_dim=(1,1), use_batchnorm=True, use_xavier=True, max_channels=40, max_pool = False):
        super().__init__()
#         channel_dims = np.linspace(1, max_channels, n_hidden_layers + 1, dtype=int)
        channel_dims = np.concatenate([np.array([1]), np.repeat(max_channels, n_hidden_layers+1)])
        
        #Hidden layers
        modules = []
#         image_size = [n_mfcc, ceps_length] # inital image size
        image_size = [513, 431]
        for i in range(n_hidden_layers):
          # print(image_size)
          dim0 = dim[0]
          dim1 = dim[1]
          stride0 = stride[0]
          stride1 = stride[1]
          modules.append(nn.Conv2d(channel_dims[i], channel_dims[i+1], (dim0,dim1), stride))
          image_size[0] = int((image_size[0] - dim0) / stride0 + 1) #Image size after the convolution
          image_size[1] = int((image_size[1] - dim1) / stride1 + 1)
            
          #Batch norm
          if use_batchnorm:
              modules.append(nn.BatchNorm2d(channel_dims[i+1]))
          
          #Activation
          modules.append(activation)
          
          #Max pool
          if max_pool:
            pool_dim0 = pool_dim[0]
            pool_dim1 = pool_dim[1]
            pad0 = 0
            pad1 = 0
            pool_stride0 = pool_stride[0]
            pool_stride1 = pool_stride[1]
            modules.append(nn.MaxPool2d((pool_dim0, pool_dim1), (pool_stride0, pool_stride1), (pad0, pad1)))
            image_size[0] = int((image_size[0]+2*pad0 - pool_dim0) / pool_stride0 + 1) #Image size after max pooling
            image_size[1] = int((image_size[1]+2*pad1 - pool_dim1) / pool_stride1 + 1)
        # print(image_size)
            
        #Dropout
        # if dropout > 0 and dropout < 1:
        #     modules.append(nn.Dropout2d(dropout))
        # print(image_size)
        #Last layer
        modules.append(nn.Flatten())
        input_dim  = max_channels*image_size[0]*image_size[1]
        output_dim = np.min([int(input_dim/2), 50])
        modules.append(nn.Linear(input_dim, output_dim))
        modules.append(activation)
        modules.append(nn.Linear(output_dim, 10))
        modules.append(torch.nn.Softmax())
        
        #Concatenate
        self.layers = nn.Sequential(*modules)
        
        #Initialize weights
        if use_xavier:
            self.layers.apply(self.init_weights)
        
    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)


# In[16]:


CNN(n_hidden_layers=5, activation=nn.ReLU(), stride=(1,1), dim=(3,3), pool_stride=(2,2), pool_dim=(2,2), use_batchnorm=True, use_xavier=True, max_channels=64, max_pool = True)


# In[ ]:


#Determine whether CNN is working

#High level inputs
use_gpu         = False
seed            = 0

#Training inputs
batch_size      = 10
n_epochs        = 1
step_size       = 0.01
momentum        = 0.9

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

np.random.seed(seed)
torch.manual_seed(seed)

#Network architecture inputs
n_hidden_layers = 3
use_batchnorm   = True
use_xavier      = True
net = CNN(n_hidden_layers=n_hidden_layers, activation=nn.ReLU(), stride=(1,1), dim=(3,3), pool_stride=(2,2), 
          pool_dim=(2,2), use_batchnorm=True, use_xavier=True, max_channels=12, max_pool = False)

#Visualization inputs
show_loss_every = 20
net = train(net, n_epochs, trainloader, use_gpu, step_size, momentum, show_loss_every)
test(net, testloader, use_gpu)


# In[ ]:


for name, param in net.named_parameters():
    print(name)
    print(param)


# In[ ]:


#Train completely

#High level inputs
use_gpu         = False
seed            = 0

#Training inputs
batch_size      = 8
n_epochs        = 100
step_size       = s_selected
momentum        = m_selected

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

np.random.seed(seed)
torch.manual_seed(seed)

#Network architecture inputs
n_hidden_layers = 3
use_batchnorm   = True
use_xavier      = True
net = CNN(n_hidden_layers=n_hidden_layers, activation=a_selected, stride=(ks_selected,ks_selected), dim=(kd_selected,kd_selected), pool_stride=(1,1), pool_dim=(2,2), use_batchnorm=True, use_xavier=True, max_channels=c_selected, max_pool = True)

#Visualization inputs
show_loss_every = 50
net = train(net, n_epochs, trainloader, use_gpu, step_size, momentum, show_loss_every)
test(net, testloader, use_gpu)


# In[ ]:


# batch_size      = 8
# trainloader     = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
# testloader      = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

# n_epochs        = 3
# step_size       = np.random.choice(np.logspace(-3, 0, 100), 100)
# momentum        = np.random.random(100)

# #Fixed network architecture inputs
# n_hidden_layers = 3
# use_batchnorm   = True
# use_xavier      = True

# #Hyperparameters
# activation = np.array([nn.ReLU(), nn.LeakyReLU()])
# max_channels = np.arange(5, 100, 5)
# kernel_dim = np.array([2,3])
# kernel_stride = np.array([1,2])

# num_search = 20
# a_search = activation[np.random.randint(0,len(activation),num_search)]
# c_search = max_channels[np.random.randint(0,len(max_channels),num_search)]
# s_search = step_size[np.random.randint(0,len(step_size), num_search)]
# m_search = momentum[np.random.randint(0,len(momentum), num_search)]
# kd_search = kernel_dim[np.random.randint(0,len(kernel_dim), num_search)]
# ks_search = kernel_stride[np.random.randint(0,len(kernel_stride), num_search)]

# val_accuracy = np.empty(num_search)
# for i in range(num_search):
#   a = a_search[i]
#   c = c_search[i]
#   s = s_search[i]
#   m = m_search[i]
#   kd = kd_search[i]
#   if kd == 3: ks = 1
#   else: ks = ks_search[i]
#   print()
#   print('Activation: ', str(a), ', max channels: ', str(c), ', momentum: ', str(m), ', step size: ', str(s), ', kernel dim: ', str(kd), ', kernel stride: ', str(ks))
#   net = CNN(n_hidden_layers=n_hidden_layers, activation=a, stride=(ks,ks), dim=(kd,kd), pool_stride=(1,1), pool_dim=(2,2), use_batchnorm=True, use_xavier=True, max_channels=c, max_pool = True)
#   net = train(net, n_epochs, trainloader, use_gpu, s, m, show_loss_every=50)
#   val_accuracy[i] = test(net, testloader, use_gpu) 


# Ex. of validation:
# Activation:  ReLU() , max channels:  40 , momentum:  0.5938300983792906 , step size:  0.021544346900318846 , kernel dim:  2 , kernel stride:  1
# Accuracy: 7 %
# 
# Activation:  ReLU() , max channels:  5 , momentum:  0.1377248457779756 , step size:  0.12328467394420659 , kernel dim:  2 , kernel stride:  2
# Accuracy: 8 %
# 
# Activation:  LeakyReLU(negative_slope=0.01) , max channels:  30 , momentum:  0.5376233556508321 , step size:  0.01873817422860384 , kernel dim:  2 , kernel stride:  2
# Accuracy: 18 %
# 
# Activation:  ReLU() , max channels:  55 , momentum:  0.6124252490967343 , step size:  0.026560877829466867 , kernel dim:  2 , kernel stride:  2
# Accuracy: 18 %
# 
# Activation:  ReLU() , max channels:  60 , momentum:  0.46492216869766234 , step size:  0.04328761281083059 , kernel dim:  2 , kernel stride:  2
# Accuracy: 20 %
# 
# Activation:  ReLU() , max channels:  35 , momentum:  0.46492216869766234 , step size:  0.010722672220103232 , kernel dim:  2 , kernel stride:  2
# Accuracy: 18 %
# 
# Activation:  ReLU() , max channels:  10 , momentum:  0.771146097054588 , step size:  0.7054802310718645 , kernel dim:  2 , kernel stride:  2
# Accuracy: 7 %
# 
# Activation:  LeakyReLU(negative_slope=0.01) , max channels:  70 , momentum:  0.46492216869766234 , step size:  0.0016297508346206436 , kernel dim:  2 , kernel stride:  1
# Accuracy: 22 %

# In[ ]:


# i_selected = np.argmax(val_accuracy)
# # a_selected = nn.ReLU() #a_search[i_selected]
# # c_selected = c_search[i_selected]
# # s_selected = s_search[i_selected]
# # m_selected = m_search[i_selected]
# # kd_selected = kd_search[i_selected]
# # ks_selected = ks_search[i_selected]

# a_selected = nn.LeakyReLU() #a_search[i_selected]
# c_selected = 55
# s_selected = 0.0023101297000831605
# m_selected = 0.1372513505241385
# kd_selected = 3
# ks_selected = 1
# # LeakyReLU(negative_slope=0.01) , max channels:  55 , momentum:  0.1372513505241385 , step size:  0.0023101297000831605 , kernel dim:  3 , kernel stride:  1


# In[ ]:


print(prediction)

