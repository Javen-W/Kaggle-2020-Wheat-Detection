import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import WheatDataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(777)

# Hyperparameters
batch_size = 8
n_workers = 4
n_epochs = 10
lr = 0.005

def collate_fn(x):
    return tuple(zip(*x))

# Load datasets
train_dataset = WheatDataset(mode='train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, collate_fn=collate_fn)

val_dataset = WheatDataset(mode='val')
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, collate_fn=collate_fn)

# Load model
model_weights = "models/fasterrcnn_resnet50_fpn_COCO-V1.pt"
model = fasterrcnn_resnet50_fpn(
    weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
    progress=True,
    num_classes=2,  # 1 class (wheat) + background
)
if os.path.exists(model_weights):
    model.load_state_dict(torch.load(model_weights, weights_only=True))
model.to(device)

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

# Train model
for epoch in range(n_epochs):
    model.train()
    for batch in tqdm(train_loader, leave=False, desc=f"Train epoch {epoch+1}/{n_epochs}"):
        # Extract batch items
        images, targets = batch
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward-feed and back-propagate
        loss_dict = model(images, targets) # Returns losses
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {losses.item()}")

# Save model
torch.save(model.state_dict(), model_weights)
