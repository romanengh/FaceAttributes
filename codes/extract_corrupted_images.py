from IPython.display import display
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Path to attributes file
attr_file = os.path.join(PROJECT_ROOT,"list_attr_celeba.txt")
image_dir = os.path.join(PROJECT_ROOT,"img_align_celeba") #cleaned file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing : Detection and isolation of corrupted images

# Configuration

images_file = image_dir
corrupted_file = os.path.join(PROJECT_ROOT,"corrupted_images")

if not os.path.exists(corrupted_file):
    os.makedirs(corrupted_file)

print(f"Running 'Black and white detection' analysis in {images_file}...")

files = [f for f in os.listdir(images_file) if f.endswith(('.jpg', '.png'))]
count_suspects = 0

for image_name in tqdm(files):
    image_path = os.path.join(images_file, image_name)

    # We read the image in color to test if it has any

    img = cv2.imread(image_path)

    # If the file is broken (unreadable), we move it to the corrupted images folder
    if img is None: 
        shutil.move(image_path, os.path.join(corrupted_file, image_name))
        count_suspects += 1
        continue

    # We convert to HSV (Hue, Saturation, Value). The 'S' channel (saturation) tells us if the image is colored or gray.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1] # we only keep the saturation channel

    """ Now we calculate the mean saturation. 
        A color image has a mean > 20-30, while a B&W image has a mean close to 0."""
    
    mean_saturation = np.mean(saturation)

    # If the mean saturation is below 10, we consider that the image is B&W.
    if mean_saturation < 10:
        shutil.move(image_path, os.path.join(corrupted_file, image_name))
        count_suspects += 1

print("-" * 30)
print(f"Finished ! {count_suspects} images (B&W or gray) moved to file {corrupted_file}.")