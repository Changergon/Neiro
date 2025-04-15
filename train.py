import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random

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
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3),
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
    train_dataset = ImageFolder(train_dir, transform=get_transforms(train=True), loader=safe_pil_loader)
    val_dataset = ImageFolder(val_dir, transform=get_transforms(train=False), loader=safe_pil_loader)
    check_dataset = ImageFolder(check_dir, transform=get_transforms(train=False), loader=safe_pil_loader)
    return train_dataset, val_dataset, check_dataset

# --- Модель ---
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Обучение (с MixUp или CutMix) ---
def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=True):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(loader, desc="Обучение"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


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

# --- По классам ---
def evaluate_per_class(model, loader, class_names, device):
    model.eval()
    correct = [0] * len(class_names)
    total = [0] * len(class_names)
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Проверка"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for label, pred in zip(labels, preds):
                total[label] += 1
                if label == pred:
                    correct[label] += 1

    results = []
    for i, class_name in enumerate(class_names):
        acc = 100 * correct[i] / total[i] if total[i] > 0 else 0
        results.append((class_name, acc))

    for class_name, acc in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{class_name}: {acc:.2f}%")

# --- Матрица ошибок ---
def plot_confusion_matrix(model, loader, class_names, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues')
    plt.title("Confusion Matrix on Check Set")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    print("📉 Матрица ошибок сохранена в confusion_matrix.png")

# --- Ошибки ---
def save_misclassified_examples(model, dataset, loader, device, save_dir="misclassified"):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    true_class = dataset.classes[labels[i]]
                    pred_class = dataset.classes[preds[i]]
                    folder = os.path.join(save_dir, f"{true_class}_as_{pred_class}")
                    os.makedirs(folder, exist_ok=True)
                    img_path, _ = dataset.samples[batch_idx * loader.batch_size + i]
                    shutil.copy(img_path, os.path.join(folder, os.path.basename(img_path)))

    print(f"❌ Ошибочно классифицированные изображения сохранены в папке '{save_dir}'")

# --- Основной цикл ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Используемое устройство:", device)
    writer = SummaryWriter("runs/train_logs")

    remove_empty_folders(args.train_dir)
    remove_empty_folders(args.val_dir)

    train_dataset, val_dataset, check_dataset = load_datasets(args.train_dir, args.val_dir, args.check_dir)
    class_counts = np.array([len(os.listdir(os.path.join(args.train_dir, class_name))) for class_name in train_dataset.classes])
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[train_dataset.targets[i]] for i in range(len(train_dataset))])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    check_loader = DataLoader(check_dataset, batch_size=args.batch_size)
    num_classes = len(train_dataset.classes)

    model = CustomCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=5)

    best_accuracy = 0.0
    losses, accuracies = [], []

    for epoch in range(args.epochs):
        print(f"\n--- Эпоха {epoch+1}/{args.epochs} ---")
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, use_mixup=True)
        accuracy = evaluate(model, val_loader, device)
        print(f"Потери: {avg_loss:.4f}, Вал. Точность: {accuracy:.2f}%")

        losses.append(avg_loss)
        accuracies.append(accuracy)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/val", accuracy, epoch)

        scheduler.step(accuracy)
        early_stopping(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model, args.model_path)  # сохраняем всю модель
            print(f"✅ Лучшая модель сохранена ({best_accuracy:.2f}%)")

        if early_stopping.early_stop:
            print("⏹️ Ранняя остановка: точность не улучшалась несколько эпох подряд.")
            break

    # --- Графики ---
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Потери')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Точность')
    plt.legend()
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/train_stats.png")
    print("📈 Графики сохранены в train_stats.png")

    # --- Проверка ---
    print("\n--- Проверка на контрольном наборе (check) ---")
    model = torch.load(args.model_path, weights_only=True)
  # загружаем всю модель
    model.to(device)
    total_check_acc = evaluate(model, check_loader, device)
    print(f"\n📊 Общая точность на контрольном наборе: {total_check_acc:.2f}%")
    evaluate_per_class(model, check_loader, check_dataset.classes, device)
    plot_confusion_matrix(model, check_loader, check_dataset.classes, device)
    save_misclassified_examples(model, check_dataset, check_loader, device)

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default="C:/Users/Дмитрий/Desktop/ECOMMERCE_PRODUCT_IMAGES/train")
    parser.add_argument('--val_dir', type=str, default="C:/Users/Дмитрий/Desktop/ECOMMERCE_PRODUCT_IMAGES/val")
    parser.add_argument('--check_dir', type=str, default="C:/Users/Дмитрий/Desktop/ECOMMERCE_PRODUCT_IMAGES/check")
    parser.add_argument('--model_path', type=str, default="best_model.pth")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0005)
    args = parser.parse_args()
    main(args)