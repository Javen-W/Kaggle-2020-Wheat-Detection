import pandas
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import WheatDataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
import matplotlib as plt
import matplotlib.patches as patches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(777)

# Hyperparameters
batch_size = 8
n_workers = 4
n_epochs = 1
lr = 0.005

def collate_fn(x):
    return tuple(zip(*x))

def format_prediction_string(boxes, scores):
    """Format predictions as 'score x_min y_min width height'."""
    pred_strings = []
    for box, score in zip(boxes, scores):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        pred_strings.append(f"{score:.4f} {x_min:.2f} {y_min:.2f} {width:.2f} {height:.2f}")
    return " ".join(pred_strings) if pred_strings else ""

# Load datasets
train_dataset = WheatDataset(mode='train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, collate_fn=collate_fn)

val_dataset = WheatDataset(mode='val')
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, collate_fn=collate_fn)

test_dataset = WheatDataset(mode='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=n_workers, collate_fn=collate_fn)

# Load model
model_weights = "models/fasterrcnn_resnet50_fpn_COCO-V1.pt"
model = fasterrcnn_resnet50_fpn(
    weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
    progress=True,
)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, num_classes=2) # 1 class (wheat) + background

# Load custom weights if they exist
if os.path.exists(model_weights):
    model.load_state_dict(torch.load(model_weights, weights_only=True))
model.to(device)

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

# Training loop
for epoch in range(n_epochs):
    # Train
    model.train()
    train_loss = 0.0
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
        train_loss += losses.item()
    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss / len(train_loader):.4f}")

    # Validate
    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[0.5 + i * 0.05 for i in range(6)])  # 0.5 to 0.75
    with torch.no_grad():
        for batch in tqdm(val_loader, leave=False, desc=f"Val epoch {epoch + 1}/{n_epochs}"):
            # Extract batch items
            images, targets = batch
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Evaluate
            outputs = model(images)
            preds = [
                {
                    'boxes': output['boxes'],
                    'scores': output['scores'],
                    'labels': output['labels'],
                } for output in outputs
            ]
            metric.update(preds, targets)
    mAP = metric.compute()['map'].item()
    print(f"Epoch {epoch + 1}/{n_epochs}, Val: {mAP:.4f}")

# Save model
torch.save(model.state_dict(), model_weights)

# Inference step for test dataset
def test_predict(model, test_loader, output_file="submission.csv"):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            # Extract batch items
            images, targets = batch
            images = [image.to(device) for image in images]
            # Predict
            outputs = model(images)
            for output, target in zip(outputs, targets):
                image_id = test_dataset.image_ids[target['image_id'].item()]
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                # Filter
                mask = scores > 0.5
                boxes = boxes[mask]
                scores = scores[mask]
                pred_str = format_prediction_string(boxes, scores)
                predictions.append({'image_id': image_id, 'PredictionString': pred_str})

    # Save predictions
    submission_df = pandas.DataFrame(data=predictions)
    submission_df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}, len={len(submission_df)}")

def visualize_predictions(image, boxes, scores, threshold=0.5):
    fig, ax = plt.subplots(1)
    image = image.permute(1, 2, 0).numpy() * 255.0
    ax.imshow(image.astype(np.uint8))
    for box, score in zip(boxes, scores):
        if score > threshold:
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min, f'{score:.2f}', color='white', fontsize=8, backgroundcolor='red')
    plt.show()

test_predict(model, test_loader)  # output_file="/kaggle/working/submission.csv"



