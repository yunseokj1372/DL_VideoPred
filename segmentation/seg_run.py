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
import seg
import torchmetrics
import gc  




jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
mask = np.load("./data/Dataset_Student/train/video_0/mask.npy")

model=deeplab_res50(num_classes=49, weights=None, backbone_weights=None)
#criterion = nn.CrossEntropyLoss(weight=back_weights_prop(49,100))
criterion = nn.CrossEntropyLoss(weight=seg.back_weights_prop(49,100))
batch_size=32 #changed from 8
num_epochs=10
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.01)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
gc.collect()


def evaluation(model):
    seg.train_model_outer(100,10, model, device=device, beg=0, num_epochs=num_epochs, batch_size=batch_size,criterion=criterion, optimizer=optimizer, scheduler=scheduler)
    model.eval()
    out = model(seg.transform_image("./data/Dataset_Student/train/video_0/image_21.png").unsqueeze(0))['out'][0]
    print(jaccard(out.argmax(0),torch.tensor(mask[-1])))
    # plt.imshow(out.argmax(0))



def main():
    evaluation(model)


if __name__ == '__main__':
    main()
