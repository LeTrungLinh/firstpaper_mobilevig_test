import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Normalize, ToTensor
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
import os 

def preprocess_image(img: np.ndarray, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([ToTensor(), Normalize(mean=mean, std=std)])
    return preprocessing(img.copy()).unsqueeze(0)

def deprocess_image(img):
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    img = img * 0.1 + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

def visualize_feature(img_file , model, target_layers):
    """ Visulize feature using gradcam
        Function result will save image in same folder with image file
    Args:
        img_file (str): path to image file - "./image_test/demo.jpg"
        model (object): model object
        target_layers (_type_): final layer of backbone extract the feature of model
    """
    targets = None

    # Read image and preprocess
    rgb_img = cv2.imread(img_file, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # visualize feature using gradcam
    cam_algorithm = GradCAM
    with cam_algorithm(model=model, target_layers=target_layers, use_cuda=False) as cam:
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        base_name, ext = os.path.splitext(img_file)
        if ext in ['.jpg', '.png', '.jpeg']:
            new_img_file = f"{base_name}_result{ext}"
            cv2.imwrite(new_img_file, cam_image)
            print("Successfull save image")
        else:
            print("Invalid file extension")
