import os
import torch
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50 as deeplab_res50
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large as deeplab_mobilenet
from torchvision.models.segmentation import deeplabv3_resnet101 as deeplab_res101
from torchvision.models.segmentation import fcn_resnet50 as fcn_res50
from torchvision.models.segmentation import fcn_resnet101 as fcn_res101
from torchvision.models.segmentation import lraspp_mobilenet_v3_large as lraspp
import torch.optim as optim

import torchmetrics
jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
import gc


def transform_image(image):
  
  image = Image.open(image)
  transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.GaussianBlur(3,0.5)
    ])
  
  image = image.convert("RGB")
  image = transform(image)

  return image

def download_data(num_folders, start=0):

  train_x=[]
  train_y=[]

  folder_count=0
  for i in os.listdir("../data/Dataset_Student/train/")[start:]:

    #increment number of videos we have grabbed
    folder_count+=1

    #load mask for these frames
    mask=np.load(f'../data/Dataset_Student/train/{i}/mask.npy')

    for j in range(22):

      train_x.append(torch.tensor(transform_image(f'../data/Dataset_Student/train/{i}/image_{j}.png'),dtype=torch.float))

      labels=[]
      masky=mask[j].flatten()
      for k in range(49):
        emp = np.zeros(masky.shape[0])
        inds = np.where(masky==k)
        emp[inds]+=1
        labels.append(torch.tensor(emp.reshape((160,240)),dtype=torch.float))

      labels=torch.stack(labels)
      train_y.append(labels)

    if folder_count==num_folders:
      break
  
  train_x = torch.stack(train_x)
  train_y = torch.stack(train_y)

  return train_x, train_y  

def train_deeplabv3(inputs, labels, num_epochs, batch_size, device, model, criterion, optimizer, scheduler):

    # Move model to device
    # model.to(device)
    model.train()
    inputs.to(device)
    labels.to(device)

    # Train the model
    for epoch in range(num_epochs):
        
        running_loss = 0.0
        i=0
        while i < inputs.shape[0]:

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs[i:i+batch_size])['out']

            #dim for both should be batch_size*48*160*240
            loss = criterion(outputs, labels[i:i+batch_size])

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            i+=batch_size

        print(f'epoch {epoch+1} loss: {running_loss}')
    scheduler.step()

def train_model_outer(num_outer_batch, outer_batch_size, model,device, criterion, optimizer, scheduler, beg=0,num_epochs = 6, batch_size = 4):

  for i in range(num_outer_batch):

    train_x, train_y = download_data(outer_batch_size,beg)
    train_deeplabv3(inputs=train_x, labels=train_y, num_epochs=num_epochs, batch_size=batch_size, device=device, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler)

    beg+=outer_batch_size
    print(f'trained outer batch {i+1}')

class DiceLoss(nn.Module):
	'''Dice Loss (F-score, ...)'''

	def __init__(self, smooth=1.):
		super().__init__()

		self.smooth = smooth

	def forward(self, outputs, targets):
		inter = (outputs * targets).sum()
		dice = (2. * inter + self.smooth) / (outputs.sum() + targets.sum() + self.smooth)

		return 1. - dice


class IOULoss(nn.Module):
	'''Intersection Over Union Loss'''

	def __init__(self, smooth=1.):
		super().__init__()

		self.smooth = smooth

	def forward(self, outputs, targets):
		inter = (outputs * targets).sum()
		union = outputs.sum() + targets.sum() - inter
		iou = (inter + self.smooth) / (union + self.smooth)

		return 1. - iou

class dummy:

  def init():
    pass

  @staticmethod
  def step():
    pass

def back_weights_prop(n_classes, mult):
  """create weighting for cross entropy loss"""

  out=np.zeros(n_classes)
  out[1:]=mult
  out[0]=1

  return torch.tensor(out/sum(out))