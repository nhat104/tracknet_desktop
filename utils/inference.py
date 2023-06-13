import torch
import torchvision
import numpy as np
import cv2


def inference(model, img, image_size=(360, 640)):
    h, w = img.shape[:2]
    frames_torch = []
    frame_torch = torch.tensor(img).permute(2, 0, 1).float() / 255
    frame_torch = torchvision.transforms.functional.resize(
        frame_torch, image_size, antialias=True
    )
    frames_torch.append(frame_torch)
    frames_torch = torch.cat(frames_torch, dim=0).unsqueeze(0)
    pred = model(frames_torch)
    pred = pred[0, :, :, :].detach().cpu().numpy()

    pred_frame = pred[0, :, :]
    y, x = np.where(pred_frame == np.max(pred_frame))
    x, y = x[0], y[0]
    center = (int(x / image_size[1] * w), int(y / image_size[0] * h))

    cv2.circle(img, center, 5, (0, 255, 0), 2)
    return img
