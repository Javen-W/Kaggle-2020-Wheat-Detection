import torch
import os
from torch.utils.data import DataLoader
from src.dataset import WheatDataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn

device = torch.device('cuda')

def collate_fn(x):
    return tuple(zip(*x))

# Load datasets
train_dataset = WheatDataset(mode='train')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)

# Load models
model_weights = "models/fasterrcnn_resnet50_fpn.pt"
model = fasterrcnn_resnet50_fpn(
    progress=True,
    num_classes=1,
)
if os.path.exists(model_weights):
    model.load_state_dict(torch.load(model_weights, weights_only=True))
model.to(device)

model.train()
for batch in train_loader:
    continue

torch.save(model.state_dict(), model_weights)
