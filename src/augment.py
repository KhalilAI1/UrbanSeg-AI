import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image

def get_augmentation_pipeline():
    transform = A.Compose([
        A.Resize(512, 512),  # Resize to a fixed size
        A.RandomCrop(width=256, height=256, p=1.0),
        
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5), 
        A.Rotate(limit=15, p=0.5, border_mode=0), 
        
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return transform

if __name__ == "__main__":
    # Example usage
    transform = get_augmentation_pipeline()
    img_np = cv2.imread("data\\row\\train_images\\2007_000032.jpg")
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # Convert to RGB
    mask_np = Image.open("data\\row\\train_labels\\2007_000032.png")  # Load mask in original mode
    mask_np = np.array(mask_np, dtype=np.int64)  # Convert to numpy array (preserves class indices)
    
    out = transform(image=img_np, mask=mask_np)
    print("Image:", out['image'].shape, out['image'].dtype)
    print("Mask:", out['mask'].shape, out['mask'].dtype, np.unique(out['mask']))