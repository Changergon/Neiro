import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
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

# --- Инициализация модели ---
def create_model(num_classes, device):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

# --- Обучение одной эпохи ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(loader, desc="Обучение"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
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
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# --- Проверка по категориям ---
def evaluate_per_class(model, loader, class_names, device):
    model.eval()
    correct = [0] * len(class_names)
    total = [0] * len(class_names)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for label, pred in zip(labels, preds):
                total[label] += 1
                if label == pred:
                    correct[label] += 1
    for i, class_name in enumerate(class_names):
        acc = 100 * correct[i] / total[i] if total[i] > 0 else 0
        print(f"{class_name}: {acc:.2f}%")

# --- Основной цикл ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Используемое устройство:", device)

    remove_empty_folders(args.train_dir)
    remove_empty_folders(args.val_dir)

    train_dataset, val_dataset, check_dataset = load_datasets(args.train_dir, args.val_dir, args.check_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    check_loader = DataLoader(check_dataset, batch_size=args.batch_size)
    num_classes = len(train_dataset.classes)

    model = create_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    best_accuracy = 0.0
    patience_counter = 0
    early_stopping_patience = 4
    losses, accuracies = [], []

    for epoch in range(args.epochs):
        print(f"\n--- Эпоха {epoch+1}/{args.epochs} ---")
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        accuracy = evaluate(model, val_loader, device)
        print(f"Потери: {avg_loss:.4f}, Вал. Точность: {accuracy:.2f}%")

        losses.append(avg_loss)
        accuracies.append(accuracy)

        scheduler.step(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), args.model_path)
            print(f"✅ Лучшая модель сохранена ({best_accuracy:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
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
    plt.savefig("train_stats.png")
    print("Графики сохранены в train_stats.png")

    # --- Проверка по check-набору ---
    print("\n--- Проверка на контрольном наборе (check) ---")
    model.load_state_dict(torch.load(args.model_path))
    evaluate_per_class(model, check_loader, check_dataset.classes, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default="C:/Users/Дмитрий/Desktop/ECOMMERCE_PRODUCT_IMAGES/train")
    parser.add_argument('--val_dir', type=str, default="C:/Users/Дмитрий/Desktop/ECOMMERCE_PRODUCT_IMAGES/val")
    parser.add_argument('--check_dir', type=str, default="C:/Users/Дмитрий/Desktop/ECOMMERCE_PRODUCT_IMAGES/check")
    parser.add_argument('--model_path', type=str, default="best_model.pth")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.0005)
    args = parser.parse_args()
    main(args)
