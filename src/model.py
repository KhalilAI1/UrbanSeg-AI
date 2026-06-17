import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import torch

def get_model(num_classes, pretrained = True):
    if pretrained:
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=weights, weights_backbone=None)
    else:
        model = deeplabv3_resnet50(weights=None, weights_backbone=None)

    # Replace the classifier head to match our number of classes
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)


    # Remove auxiliary classifier for simplicity and save VRAM 
    model.aux_classifier = None  

    return model


if __name__ == "__main__":
    model = get_model(num_classes=3, pretrained=True)
    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
    print(type(output))
    print(output['out'].shape)
