import torch
import torchvision
import numpy as np
import cv2 as cv


def inference(model, img, image_size=(360, 640)):
    rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    image = torchvision.transforms.ToTensor()(rgb_image)
    image = torchvision.transforms.Resize(size=image_size, antialias=True)(image)
    image = image.type(torch.float32)
    image = image.unsqueeze(0)

    pred = model(image)
    pred_frame = pred[0, 0]
    pred_frame = pred_frame.detach().numpy()

    y, x = np.where(pred_frame == np.max(pred_frame))
    x, y = x[0], y[0]

    h, w = img.shape[:2]
    center = (int(x / image_size[1] * w), int(y / image_size[0] * h))
    cv.circle(img, center, 5, (0, 255, 0), 2)
    return img
