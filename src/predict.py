import torch
from PIL import Image
import numpy as np 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import get_model



def predict_mask(img, model, device):
    # Load the trained model
    model.eval()
    if isinstance(img, np.ndarray):
        image = Image.fromarray(img).convert("RGB")
    else:
        image = Image.open(img).convert("RGB")

    # Load and preprocess the test image
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    image_np = np.array(image)

    transformed = transform(image=image_np)
    input_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension

    # Predict the mask
    with torch.no_grad():
        output = model(input_tensor.to(device))
        predicted_mask = torch.argmax(output['out'], dim=1).squeeze().cpu().numpy()

    return predicted_mask

def colorize_mask (mask):

    color_map = np.array([
    [0, 0, 0],      # 0: background
    [255, 0, 0],    # 1: person
    [0, 255, 0],    # 2: vehicle
    [255, 255, 255] # 255: ignore
    ], dtype=np.uint8)
    mask = mask.copy()
    mask[mask == 255] = 0
    color_mask = color_map[mask]
    return color_mask

def overly (image, mask, alpha=0.5):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")

    color_mask = colorize_mask(mask)
    color_mask_pil = Image.fromarray(color_mask)
    color_mask_size = color_mask_pil.resize(image.size, Image.NEAREST)
    overlay = (1 - alpha) * np.array(image) + alpha * np.array(color_mask_size)
    return Image.fromarray(overlay.astype(np.uint8))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the trained model
    model = get_model(num_classes=3, pretrained=True)
    model.to(device)
    model.load_state_dict(torch.load("outputs/weights/best_model.pth", map_location=device))
    # Predict the mask for the test image
    predicted_mask = predict_mask("outputs\\results\\2007_005173.jpg", model, device)
    # Create an overlay of the original image and the predicted mask
    overlay_image = overly(Image.open("outputs\\results\\2007_005173.jpg").convert("RGB"), predicted_mask)
    overlay_image.save("outputs/results/overlay.png")
    # Save the colorized mask
    color_mask = colorize_mask(predicted_mask)
    result_image = Image.fromarray(color_mask)
    result_image.save("outputs/results/mask.png")