import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import torchmetrics
import sys

jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)


def transform_image(image):

    image = Image.open(image)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.GaussianBlur(3, 0.5),
        ]
    )

    image = image.convert("RGB")
    image = transform(image)

    return image


def back_weights_prop(n_classes, mult):
    """create weighting for cross entropy loss"""

    out = np.zeros(n_classes)
    out[1:] = mult
    out[0] = 1

    return torch.tensor(out / sum(out))


def download_data(num_folders, directory, start=0):

    train_x = []
    train_y = []

    folder_count = 0
    for i in os.listdir(directory)[start:]:

        # increment number of videos we have grabbed
        folder_count += 1

        # load mask for these frames
        mask = np.load(f"{directory}/{i}/mask.npy")

        for j in range(22):

            train_x.append(
                torch.tensor(
                    transform_image(f"{directory}/{i}/image_{j}.png"), dtype=torch.float
                )
            )

            labels = []
            masky = mask[j].flatten()
            for k in range(49):
                emp = np.zeros(masky.shape[0])
                inds = np.where(masky == k)
                emp[inds] += 1
                labels.append(torch.tensor(emp.reshape((160, 240)), dtype=torch.float))

            labels = torch.stack(labels)
            train_y.append(labels)

        if folder_count == num_folders:
            break

    train_x = torch.stack(train_x)
    train_y = torch.stack(train_y)

    return train_x, train_y


def get_validation_error(
    model_path, model, validation_folder_path, criterion, device, num_test_images
):

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    criterion.to(device)
    mean_criterion_model = 0.0
    mean_criterion_ground = 0.0
    tested = 0

    for i in range(len(os.listdir(validation_folder_path))):

        train_x, train_y = download_data(1, validation_folder_path, i)
        train_x = train_x.unsqueeze(0).to(device)
        train_y = train_y.to(device)

        for j in range(len(train_x)):

            model_output = model(train_x[j])["out"][0]
            mean_criterion_model += criterion(model_output, train_y[j])
            mean_criterion_ground += criterion(train_y[j], train_y[j])

            tested += 1
            if tested == num_test_images:
                return mean_criterion_model / (
                    len(os.listdir(validation_folder_path)) * 22
                ), mean_criterion_ground / (
                    len(os.listdir(validation_folder_path)) * 22
                )

    return mean_criterion_model / (
        len(os.listdir(validation_folder_path)) * 22
    ), mean_criterion_ground / (len(os.listdir(validation_folder_path)) * 22)


def write_loss_to_file(filepath, loss1, loss2):

    with open(filepath, "w") as out:

        out.write(f"Loss on predicted: {loss1}")
        out.write(f"Loss on ground truth: {loss2}")


def main(model_path, model, validation_folder_path, out_path, num_test_images=100):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = jaccard

    loss_pred, loss_ground = get_validation_error(
        model_path,
        model,
        validation_folder_path,
        criterion,
        device,
        num_test_images=num_test_images,
    )

    write_loss_to_file(out_path, loss_pred, loss_ground)


if __name__ == "__main__":

    if len(sys.argv) == 6:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

else:
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
