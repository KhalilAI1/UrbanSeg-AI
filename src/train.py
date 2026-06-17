import torch
from model import get_model
from augment import get_augmentation_pipeline
from torch.utils.data import DataLoader
from dataset import UrbanSegDataset 
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics import JaccardIndex
import os 

#--- Load configuration ---
with open("config.yaml","r") as file:
    config = yaml.safe_load(file)



epochs = config.get("training", {}).get("epochs")
batch = config.get("training", {}).get("batch_size")
device = config.get("training", {}).get("device")
device = torch.device(device if torch.cuda.is_available() else "cpu")   
num_classes = config.get("data", {}).get("num_classes")

#--- Prepare datasets and dataloaders ---
train_dataset = UrbanSegDataset(config.get("data", {}).get("train_images"), config.get("data", {}).get("train_masks"), get_augmentation_pipeline())
val_transform = A.Compose([
    A.Resize(256, 256),  # Resize to a fixed size
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()  # Convert to PyTorch tensor
])
 # Validation dataset should not have random augmentations, only deterministic ones like resizing and normalization
val_dataset = UrbanSegDataset(config.get("data", {}).get("val_images"), config.get("data", {}).get("val_masks"), val_transform)
train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

#--- Prepare model, loss function, and optimizer ---
model = get_model(num_classes, pretrained=True)
model.to(device)

# Use CrossEntropyLoss which is suitable for multi-class segmentation tasks
criterion = torch.nn.CrossEntropyLoss(ignore_index=255) # ignore_index=255 to ignore void class in loss calculation

# Use different learning rates for backbone and head as specified in the config
lr_backbone = config.get("training", {}).get("lr", {}).get("lr_backbone")
lr_head = config.get("training", {}).get("lr", {}).get("lr_head")
# Set up optimizer with different learning rates for backbone and head
optimizer = torch.optim.SGD([
    {"params": model.backbone.parameters(), "lr": lr_backbone},
    {"params": model.classifier.parameters(), "lr": lr_head}
], momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=config.get("training", {}).get("scheduler", {}).get("min_lr")
)


jaccard_metric = JaccardIndex(task="multiclass",
                            num_classes=3,
                            ignore_index=255,
                            average=None  # يعطي IoU لكل class منفصلة
                            ).to(device)

best_miou = 0.0
#--- Training loop ---
for epoch in range(epochs):
    model.train()
    loss_sum = 0.0
    for image, mask in train_loader:
        image, mask = image.to(device), mask.to(device).long()
        optimizer.zero_grad()
        pred = model(image)
        loss = criterion(pred['out'], mask)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    avg_loss = loss_sum / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    #--- Validation loop ---
    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for image, mask in val_loader:
            image, mask = image.to(device), mask.to(device).long()
            
            # Compute validation loss
            outputs = model(image)['out']
            loss = criterion(outputs, mask)
            val_loss_sum += loss.item()
            # Update Jaccard metric
            pred = outputs.argmax(dim=1)
            jaccard_metric.update(pred, mask)
            
    avg_val_loss = val_loss_sum / len(val_loader)
    print(f"Epoch [{epoch+1}/{epochs}] - Validation Loss: {avg_val_loss:.4f}")
    jaccard = jaccard_metric.compute()
    print(f"Epoch [{epoch+1}/{epochs}] - Per-class IoU:")
    print(f"  Background: {jaccard[0]:.4f}")
    print(f"  Person:     {jaccard[1]:.4f}")
    print(f"  Vehicle:    {jaccard[2]:.4f}")
    print(f"  mIoU:       {jaccard.mean():.4f}")
    jaccard_metric.reset()
    scheduler.step()
    if jaccard.mean() > best_miou:
        best_miou = jaccard.mean()
        dir_path = os.path.join(config.get("checkpoint").get("save_dir"), "best_model.pth")
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)
        torch.save(model.state_dict(), dir_path)
        print(f"New best mIoU: {best_miou:.4f} - Model saved.")
