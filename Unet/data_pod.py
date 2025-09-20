import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
class SegmenDataset(Dataset):
    def __init__(self, img_dir = r"Pothole_Segmentation_YOLOv8/train/images", mask_dir = r"Pothole_Segmentation_YOLOv8/train/masks", img_transform=None, mask_transform =None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        self.mask_dict = {}
        for m in os.listdir(mask_dir):
            if m.endswith(('.png', '.jpg')):
                key = os.path.splitext(m)[0]  
                self.mask_dict[key] = os.path.join(mask_dir, m)
        self.images = []
        for f in os.listdir(img_dir):
            if f.endswith(('.png', '.jpg')):
                key = os.path.splitext(f)[0]
                if key in self.mask_dict:
                    self.images.append(os.path.join(img_dir, f))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        key = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = self.mask_dict[key]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        mask = (mask > 127).astype(np.uint8) 
        mask = Image.fromarray(mask)
        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from data_pod import SegmenDataset
    dataset = SegmenDataset()
    img, mask = dataset[10]
    print(mask.size)
    # plt.subplot(1,2,1)
    # plt.imshow(img)
    # plt.title("Image")

    # plt.subplot(1,2,2)
    # plt.imshow(mask, cmap="gray")
    # plt.title("Mask")

    # plt.show()

