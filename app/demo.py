import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import gradio as gr
import torch 
from predict import predict_mask, overly
from model import get_model



# --- Load the Model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model(num_classes=3, pretrained=False) # Set pretrained to False for inference
model.load_state_dict(torch.load("outputs/weights/best_model.pth", map_location=device)) # Load the trained model weights
model.to(device)
model.eval()  # Set the model to evaluation mode

# --- Prediction Function ---
def predict(image):

    pred_mask = predict_mask(image, model, device)
    overlayed_image = overly(image, pred_mask)
    
    return overlayed_image  # Return the overlayed image for demonstration purposes

# --- Gradio Interface ---
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Image()
)

# --- Launch the Gradio App ---
demo.launch()
