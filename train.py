import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from PIL import Image


print("CUDA доступна:", torch.cuda.is_available())
print("Имя устройства:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Нет CUDA")

# --- Параметры ---
DATA_DIR = r"C:\Users\Дмитрий\Desktop\rp2k_dataset\all"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = "model.pth"
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Удаление пустых папок ---
def remove_empty_folders(root_dir):
    removed = 0
    for dirpath, dirnames, _ in os.walk(root_dir):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            if not os.listdir(full_path):
                os.rmdir(full_path)
                removed += 1
    if removed:
        print(f"Удалено пустых папок: {removed}")

remove_empty_folders(TRAIN_DIR)
remove_empty_folders(TEST_DIR)

# --- Безопасный загрузчик ---
def safe_pil_loader(path):
    image = Image.open(path)
    if image.mode in ('P', 'RGBA'):
        image = image.convert('RGBA')
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    return image.convert('RGB')

# --- Преобразования ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Загрузка данных ---
train_dataset = ImageFolder(TRAIN_DIR, transform=transform, loader=safe_pil_loader)
test_dataset = ImageFolder(TEST_DIR, transform=transform, loader=safe_pil_loader)
NUM_CLASSES = len(train_dataset.classes)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# --- Модель ---
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# --- Обучение ---
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Эпоха {epoch+1}/{NUM_EPOCHS}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # --- Оценка ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Эпоха {epoch+1}: Потери = {avg_loss:.4f}, Точность = {accuracy:.2f}%")

torch.save(model.state_dict(), MODEL_PATH)
print(f"Модель сохранена в {MODEL_PATH}")
