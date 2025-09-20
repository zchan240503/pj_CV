from data_pod import SegmenDataset
from model_by import UNet
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score
import matplotlib.pyplot as plt
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, pos_weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Sửa ở đây: Sử dụng pos_weight trong BCE_loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
dvice = 'cuda' if torch.cuda.is_available() else 'cpu'
print(dvice)
img_transform = T.Compose([
    T.Resize((572,572)),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

mask_transform = T.Compose([
    T.Resize((560,560)),
    T.ToTensor() 
])
train_dataset = SegmenDataset(img_transform=img_transform, mask_transform=mask_transform)
val_dataset = SegmenDataset(img_dir=r"Pothole_Segmentation_YOLOv8/valid/images", mask_dir=r"Pothole_Segmentation_YOLOv8/valid/masks", img_transform=img_transform, mask_transform=mask_transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
print(f"train : {len(train_dataset)}, val: {len(val_dataset)}")
model = UNet(n_classes=1).to(device=dvice)
pos_weight = torch.tensor([1.5]).to(dvice) 
criterion = FocalLoss(alpha=0.25, gamma=1)
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
epochs = 10
iou_metric = BinaryJaccardIndex().to(dvice)
dice_metric = BinaryF1Score().to(dvice)
save_loss = []
save_val = []
ious, dices = [], []
print("Train dataset size:", len(train_dataset))
print("Val dataset size:", len(val_dataset))

for epoch in range(epochs):
    # train
    model.train()
    train_loss = 0.0
    for img, mask in train_dataloader:
        img = img.to(dvice)
        mask = mask.to(dvice).float()
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, mask)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    save_loss.append(train_loss/len(train_dataloader))
    model.eval()
    val_loss = 0.0
    val_iou, val_dict = 0.0, 0.0
    with torch.no_grad():
        for img, mask in val_dataloader:
            img = img.to(dvice)
            mask = mask.to(dvice).float()
            outputs = model(img)
            loss = criterion(outputs, mask.float())
            val_loss+=loss.item()
            # probs = torch.sigmoid(outputs)  # [B,1,H,W]
            # print("prob range:", probs.min().item(), probs.max().item())
            
            preds = (torch.sigmoid(outputs) > 0.2).float()
            mask = mask.to(dvice).float()
            mask = (mask > 0).float()
            val_iou += iou_metric(preds,mask).item()
            val_dict += dice_metric(preds, mask).item()
        save_val.append(val_loss/len(val_dataloader))    
        ious.append(val_iou/len(val_dataloader))
        dices.append(val_dict/len(val_dataloader))
        print("preds:", preds.shape, preds.unique())
        print("mask:", mask.shape, mask.unique())

    torch.save(model.state_dict(), "unet.pothole.pth")
    print(f"da luu lan {epoch+1}")
    print(f"epoch : {epoch} -- train_loss : {save_loss[-1]} -- val_loss : {save_val[-1]}")
    print(f"iou : {ious[-1]} ---- dic : {dices[-1]}")
epochs_range = range(1, epochs+1)

# --- Loss ---
plt.figure(figsize=(10,5))
plt.plot(epochs_range, save_loss, label='Train Loss')
plt.plot(epochs_range, save_val, label='Val Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# --- Metrics ---
plt.figure(figsize=(10,5))
plt.plot(epochs_range, ious, label='IoU')
plt.plot(epochs_range, dices, label='Dice')
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.title("IoU & Dice over Epochs")
plt.legend()
plt.grid(True)
plt.show()





    
