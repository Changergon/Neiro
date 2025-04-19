import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from pathlib import Path



# CBAM: Convolutional Block Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return attention


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        self.norm = nn.BatchNorm2d(in_planes)

    def forward(self, x):
        x = x * self.ca(x)  # Call self.ca as a function
        x = x * self.sa(x)  # Call self.sa as a function
        x = self.norm(x)
        return x


# DropBlock (более агрессивный аналог Dropout для ConvNet)
class DropBlock2D(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=7):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = (torch.rand_like(x[:, :, ::self.block_size, ::self.block_size]) < gamma).float()
        mask = F.interpolate(mask, size=x.shape[2:], mode='nearest')
        return x * (1 - mask)



# --- EarlyStopping ---
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# --- MixUp ---
def mixup_data(x, y, alpha=1.0):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=alpha)

    def forward(self, inputs, targets):
        logp = self.ce(inputs, targets)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

# --- Удаление пустых папок ---
def remove_empty_folders(root_dir):
    for dirpath, dirnames, _ in os.walk(root_dir):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            if not os.listdir(full_path):
                os.rmdir(full_path)

# --- Безопасный загрузчик ---
def safe_pil_loader(path):
    image = Image.open(path)
    if image.mode in ('P', 'RGBA'):
        image = image.convert('RGBA')
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    return image.convert('RGB')

# --- Трансформации ---
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.GaussianBlur(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

# --- Загрузка данных ---
def load_datasets(train_dir, val_dir, check_dir):
    train_dataset = ImageFolder(train_dir, transform=get_transforms(True), loader=safe_pil_loader)
    val_dataset = ImageFolder(val_dir, transform=get_transforms(False), loader=safe_pil_loader)
    check_dataset = ImageFolder(check_dir, transform=get_transforms(False), loader=safe_pil_loader)
    return train_dataset, val_dataset, check_dataset

# Улучшенная CNN модель
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        self.block1 = conv_block(3, 64)
        self.cbam1 = CBAM(64)
        self.drop1 = DropBlock2D(drop_prob=0.1)

        self.block2 = conv_block(64, 128)
        self.cbam2 = CBAM(128)
        self.drop2 = DropBlock2D(drop_prob=0.1)

        self.block3 = conv_block(128, 256)
        self.cbam3 = CBAM(256)
        self.drop3 = DropBlock2D(drop_prob=0.1)

        self.block4 = conv_block(256, 512)
        self.cbam4 = CBAM(512)
        self.drop4 = DropBlock2D(drop_prob=0.1)

        self.block5 = conv_block(512, 1024)
        self.cbam5 = CBAM(1024)
        self.drop5 = DropBlock2D(drop_prob=0.1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.apply_cbam_and_drop(x, 64)

        x = self.block2(x)
        x = self.apply_cbam_and_drop(x, 128)

        x = self.block3(x)
        x = self.apply_cbam_and_drop(x, 256)

        x = self.block4(x)
        x = self.apply_cbam_and_drop(x, 512)

        x = self.block5(x)
        x = self.apply_cbam_and_drop(x, 1024)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def apply_cbam_and_drop(self, x, channels):
        norm = nn.LayerNorm(x.size()[1:]).to(x.device)
        x = norm(x)

        if channels == 64:
            x = self.cbam1(x)
            if self.training:
                x = self.drop1(x)
        elif channels == 128:
            x = self.cbam2(x)
            if self.training:
                x = self.drop2(x)
        elif channels == 256:
            x = self.cbam3(x)
            if self.training:
                x = self.drop3(x)
        elif channels == 512:
            x = self.cbam4(x)
            if self.training:
                x = self.drop4(x)
        elif channels == 1024:
            x = self.cbam5(x)
            if self.training:
                x = self.drop5(x)


        return x


# --- Обучение ---
def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=True, cutmix_prob=0.0):
    model.train()
    total_loss = 0.0
    for inputs, labels in tqdm(loader, desc="Обучение"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        r = np.random.rand()
        if use_mixup and r < cutmix_prob:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        elif use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(x.device)
    y_a, y_b = y, y[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# --- Оценка ---
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Валидация"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# --- Оценка по классам ---
def evaluate_per_class(model, loader, device):
    model.eval()
    num_classes = len(loader.dataset.classes)
    all_preds = torch.zeros(len(loader.dataset)).long().to(device)
    all_labels = torch.zeros(len(loader.dataset)).long().to(device)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(loader, desc="Оценка по классам")):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds[i * loader.batch_size:(i + 1) * loader.batch_size] = preds
            all_labels[i * loader.batch_size:(i + 1) * loader.batch_size] = labels

    return confusion_matrix(all_labels.cpu(), all_preds.cpu(), labels=range(num_classes))

# --- Сохранение неверно классифицированных изображений ---
def save_misclassified_examples(model, loader, device, output_dir, num_samples=5):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    misclassified = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Сохранение неверно классифицированных"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            misclassified_mask = preds != labels
            misclassified.extend([(inputs[i], labels[i], preds[i]) for i in range(len(inputs)) if misclassified_mask[i]])

    for i, (img, label, pred) in enumerate(misclassified[:num_samples]):
        img = img.cpu().numpy().transpose((1, 2, 0)) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(f"{output_dir}/misclassified_{i}_label_{label.item()}_pred_{pred.item()}.png")

# --- Построение confusion matrix ---
def plot_confusion_matrix(conf_matrix, class_names, output_file):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(output_file)


# --- Основной цикл ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter("runs/train_logs")
    # --- Информация о системе и параметрах ---
    print(f"\n🖥️ Используемое устройство: {device}")
    if torch.cuda.is_available():
        print(f"📦 Видеокарта: {torch.cuda.get_device_name(0)}")

    print("\n🔧 Параметры запуска:")
    print(f"    📁 Train Dir:       {args.train_dir}")
    print(f"    📁 Val Dir:         {args.val_dir}")
    print(f"    📁 Check Dir:       {args.check_dir}")
    print(f"    💾 Model Path:      {args.model_path}")
    print(f"    📦 Batch Size:      {args.batch_size}")
    print(f"    🔁 Epochs:          {args.epochs}")
    print(f"    🚀 Learning Rate:   {args.lr}")
    print(f"    🎯 Loss Function:   {'Focal Loss' if args.loss == 'focal' else 'CrossEntropy'}")
    print(f"    🧪 MixUp:           {'Включен' if True else 'Отключен'}")
    print(f"    🩹 CutMix prob:     {args.cutmix_prob}")
    print(f"    ⏹️ EarlyStopping:   Patience = {args.patience}")


    remove_empty_folders(args.train_dir)
    remove_empty_folders(args.val_dir)

    train_dataset, val_dataset, check_dataset = load_datasets(args.train_dir, args.val_dir, args.check_dir)
    num_classes = len(train_dataset.classes)

    class_counts = np.array(
        [len(os.listdir(os.path.join(args.train_dir, class_name))) for class_name in train_dataset.classes])
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[train_dataset.targets[i]] for i in range(len(train_dataset))])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    check_loader = DataLoader(check_dataset, batch_size=args.batch_size)

    model = ImprovedCNN(num_classes).to(device)


    # --- Выбор функции потерь ---
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    if args.loss == 'focal':
        criterion = FocalLoss(alpha=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=5)

    best_accuracy = 0.0
    losses, accuracies = [], []

    for epoch in range(args.epochs):
        print(f"\n{'=' * 25} Эпоха {epoch + 1} / {args.epochs} {'=' * 25}")
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                   use_mixup=True, cutmix_prob=args.cutmix_prob)
        accuracy = evaluate(model, val_loader, device)

        losses.append(avg_loss)
        accuracies.append(accuracy)

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/val", accuracy, epoch)

        scheduler.step(accuracy)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("LR", current_lr, epoch)

        print(f"📉 Средняя потеря (loss):        {avg_loss:.4f}")
        print(f"✅ Точность на валидации:        {accuracy:.2f}%")
        print(f"🔁 Текущий learning rate:        {current_lr:.6f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model, args.model_path)
            print("💾 Лучшая модель сохранена!")

        if early_stopping.early_stop:
            print("⏹️ Ранняя остановка.")
            break
        print("=" * 60)

    # Оценка на check-сете по завершению всех эпох
    check_accuracy = evaluate(model, check_loader, device)
    print(f"✅ Точность на проверочном наборе (check_loader):        {check_accuracy:.2f}%")
    writer.add_scalar("Accuracy/check", check_accuracy, args.epochs)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=Path, default=Path("C:/Users/Дмитрий/Desktop/ECOMMERCE_PRODUCT_IMAGES/train"))
    parser.add_argument('--val_dir', type=Path, default=Path("C:/Users/Дмитрий/Desktop/ECOMMERCE_PRODUCT_IMAGES/val"))
    parser.add_argument('--check_dir', type=Path, default=Path("C:/Users/Дмитрий/Desktop/ECOMMERCE_PRODUCT_IMAGES/check"))
    parser.add_argument('--model_path', type=str, default="best_model.pth")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--loss', type=str, choices=['ce', 'focal'], default='focal')
    parser.add_argument('--cutmix_prob', type=float, default=0.15)
    parser.add_argument('--patience', type=int, default=5, help='Patience для EarlyStopping')
    args = parser.parse_args()

    main(args)
