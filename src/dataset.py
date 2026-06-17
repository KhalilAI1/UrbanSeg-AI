from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os
from augment import get_augmentation_pipeline


class UrbanSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        self.masks = sorted([f for f in os.listdir(masks_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        self.lookup_table = np.zeros(256, dtype=np.uint8)
        
        # Map VOC 2012 classes to our 3 classes: background (0), person (1), vehicle (2)
        self.lookup_table[15] = 1            # person → 1
        for vid in [1, 2, 4, 6, 7, 14, 19]:     # vehicles
            self.lookup_table[vid] = 2
        self.lookup_table[255] = 255         # ignore (void class)
        assert len(self.images) == len(self.masks), "Number of images and masks must match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        
        # Load the mask in its original mode to preserve class indices
        mask = Image.open(mask_path)           # Keep original palette mode
        mask = np.array(mask, dtype=np.int64)  # Convert to numpy array (preserves class indices)
        mask = self.lookup_table[mask]         # Map to our 3 classes using the lookup table


        # Apply transformations if provided
        if self.transform is not None:
            # Albumentations expects numpy arrays (H, W, C) for images
            image_np = np.array(image)
            transformed = self.transform(image=image_np, mask=mask)
            image = transformed['image']
            mask = transformed['mask']  # may be np.ndarray or torch.Tensor

        # Ensure mask is a torch LongTensor with class indices for loss/metric calculations
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        elif isinstance(mask, torch.Tensor):
            mask = mask.long()
        else:
            raise TypeError(f"Unsupported mask type: {type(mask)}")

        return image, mask

if __name__ == "__main__":
    # Example usage
    dataset = UrbanSegDataset(
    images_dir="data/row/train_images",
    masks_dir="data/row/train_labels",
    transform=get_augmentation_pipeline()
    )

    image, mask = dataset[0]
    # Print image shape/dtype compatibly for PIL/ndarray/torch.Tensor
    img_shape = tuple(image.shape)
    img_dtype = image.dtype
    mask_shape = tuple(mask.shape)
    mask_dtype = mask.dtype
    uniq = torch.unique(mask)


    print("Image:", type(image), img_shape, img_dtype)
    print("Mask:", type(mask), mask_shape, mask_dtype, uniq)