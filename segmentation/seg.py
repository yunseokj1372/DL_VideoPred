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

device='cuda'
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root, id_path, transforms = None):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(VideoDataset, self).__init__()
        self.root = root
        self.transforms = transforms
        with open(id_path, 'r') as f:
          self.ids = f.read().splitlines()

    def __getitem__(self, idx):
        id = self.ids[idx]
        img = Image.open(os.path.join(self.root, id)).convert('RGB')
        mask_path = '/'.join(id.split('/')[:2]) + '/mask.npy'
        mask_idx = int(id.split('.')[0][-1])
        mask = np.load(os.path.join(self.root, mask_path))[mask_idx]
        if self.transforms:
            img = self.transforms(img)
        return img, mask

    def __len__(self):
        return len(self.ids)

def transform_image(image):
  
  image = Image.open(image)
  transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
  
  image = image.convert("RGB")
  image = transform(image)

  return image

def load_data(start=0,outer_batch_size = 10):
  train_x=[]
  train_y=[]

  folder_count=0
  for i in os.listdir("./data/Dataset_Student/train/")[start:start+outer_batch_size]:

    #increment number of videos we have grabbed
    folder_count+=1

    #load mask for these frames
    mask=np.load(f'./data/Dataset_Student/train/{i}/mask.npy')

    for j in range(22):

      train_x.append(torch.tensor(transform_image(f'./data/Dataset_Student/train/{i}/image_{j}.png'),dtype=torch.float))

      labels=[]
      masky=mask[j].flatten()
      for k in range(49):
        emp = np.zeros(masky.shape[0])
        inds = np.where(masky==k)
        emp[inds]+=1
        labels.append(torch.tensor(emp.reshape((160,240)),dtype=torch.float))

      labels=torch.stack(labels)
      train_y.append(labels)
  
  train_x = torch.stack(train_x)
  train_y = torch.stack(train_y)
  start = start+outer_batch_size

  return train_x, train_y, start

def download_data(num_folders, start=0):

  train_x=[]
  train_y=[]

  folder_count=0
  for i in os.listdir("./data/Dataset_Student/train/")[start:]:

    #increment number of videos we have grabbed
    folder_count+=1

    #load mask for these frames
    mask=np.load(f'./data/Dataset_Student/train/{i}/mask.npy')

    for j in range(22):

      train_x.append(torch.tensor(transform_image(f'./data/Dataset_Student/train/{i}/image_{j}.png'),dtype=torch.float))

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


def train_deeplabv3_dataloader(num_epochs, batch_size, device, model, criterion, optimizer, scheduler):

    model = model.to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.GaussianBlur(3,0.5)
    ])

    train_ds = VideoDataset('./data/Dataset_Student', './train.txt', transforms=transform)
    trainloader = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)

    # Train the model
    for epoch in range(num_epochs):
      for (imgs, masks) in trainloader:

        imgs = imgs.to(device)
        masks = masks.to(device)

        min_loss = 10000
        running_loss = 0.0
        i=0
        optimizer.zero_grad()

        output = model(imgs)['out'].to(device)
        print(next(model.parameters()).device)
        print(output.device)
        print(masks.device)
        print(imgs.device)
        print(output.shape,masks.shape)
        loss = criterion(output, masks)
        

        optimizer.step()

  
        running_loss += loss.item()
        if loss< min_loss:
            min_loss = loss
            torch.save(model.state_dict(), "best_model_fcres101.pth")

        print(f'epoch {epoch+1} loss: {running_loss}')
    scheduler.step()


def train_deeplabv3(inputs, labels, num_epochs, batch_size, device, model, criterion, optimizer, scheduler):

    
    inputs = inputs.to(device)
    labels = labels.to(device)
    # Train the model
    for epoch in range(num_epochs):
     
        running_loss = 0.0
        i=0
        while i < inputs.shape[0]:

            # Zero the parameter gradients
            optimizer.zero_grad()
            print(i)
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

def get_val_labels(path):

  val_labels=[]
  for i in os.listdir(path):
    
    val_labels.append(torch.tensor(np.load(f'{path}/{i}/mask.npy')))

  return torch.stack(val_labels) 

def train_model_outer(num_outer_batch, outer_batch_size, model,device, criterion, optimizer, scheduler,direct = './path_files/best_model.pth', beg=0,num_epochs = 6, batch_size = 4):

  val_labels = get_val_labels('./data/Dataset_Student/train')
  val_labels=val_labels.to(device)
  model = model.to(device)

  print(f'val labels shape: {val_labels.size()}')
  val_iou = 0

  for i in range(num_outer_batch):
    
    model.train()
    if i ==0:
        train_x, train_y, start_new = load_data(outer_batch_size=outer_batch_size)
    else:
        train_x, train_y, start_new = load_data(start=start,outer_batch_size=outer_batch_size)
    start = start_new
    train_deeplabv3(inputs=train_x, labels=train_y, num_epochs=num_epochs, batch_size=batch_size, device=device, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler)

    model_predictions = make_segmentation_predictions(model, device, './data/Dataset_Student/val')
    print(f'model predictions shape: {model_predictions.size()}')

    val_iou_ = jaccard(model_predictions, val_labels)

    print(f'iouloss: {val_iou}')

    if val_iou_ > val_iou:

      torch.save(model.state_dict(), direct)
      val_iou=val_iou_
      print(f'new val iou: {val_iou}')


def make_segmentation_predictions(model,device, input_images_path):
    
    model.eval()
    model.to(device)
    # create predicted output tensor
    pred_output = []

    # go one video at a time
    for i in os.listdir(input_images_path):

        input = []

        for j in range(11):

            input.append(
                torch.tensor(
                    transform_image(f"{input_images_path}//{i}//image_{j}.png"),
                    dtype=torch.float,
                )
            )

        input = torch.stack(input)
        input = input.to(device)
        pred_output = pred_output + list(model(input)["out"].argmax(1))

    return torch.stack(pred_output)

    

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
