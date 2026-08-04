import torch
import torchvision.utils as vutils
import os

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

def save_sample(input_img, generated_img, target_img, path):
    images = torch.cat([input_img, generated_img, target_img], dim=0)
    vutils.save_image(images, path, normalize=True, nrow=3)
