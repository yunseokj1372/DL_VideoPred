import os
import torch
from torchvision import transforms
import sys
from PIL import Image
import torch
from torchvision.models.segmentation import deeplabv3_resnet50 as deeplab_res50
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large as deeplab_mobilenet
from torchvision.models.segmentation import deeplabv3_resnet101 as deeplab_res101
from torchvision.models.segmentation import fcn_resnet50 as fcn_res50
from torchvision.models.segmentation import fcn_resnet101 as fcn_res101
from torchvision.models.segmentation import lraspp_mobilenet_v3_large as lraspp

# hard code model class here
model = deeplab_mobilenet(num_classes=49, weights=None, weights_backbone=None)
device = "cuda"


def transform_image(image):

    image = Image.open(image)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = image.convert("RGB")
    image = transform(image)

    return image


def make_segmentation_predictions(pth, input_images_path):
    
    
    model.load_state_dict(torch.load(pth))
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

    return pred_output


if __name__ == "__main__":
    # input_images_path = './/hidden_data//hidden//' # hidden/video_16999
    
    predicted_output = make_segmentation_predictions(sys.argv[1], sys.argv[2])

    torch.save(predicted_output, sys.argv[3])
