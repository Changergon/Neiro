import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

# --- Параметры ---
DATA_DIR = r"C:\Users\Дмитрий\Desktop\rp2k_dataset\all"
UNKNOWN_DIR = os.path.join(DATA_DIR, "unknown")
CLUSTER_DIR = os.path.join(DATA_DIR, "clusters")
NEW_CLASS_DIR = os.path.join(DATA_DIR, "train")
MODEL_PATH = "model.pth"
BATCH_SIZE = 64
NUM_CLASSES = 2000
THRESHOLD = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Преобразование ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Модель и эмбеддер ---
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()
embedding_model = nn.Sequential(*list(model.children())[:-1]).to(device).eval()

# --- Извлечение эмбеддингов ---
embeddings, paths = [], []
for filename in tqdm(os.listdir(UNKNOWN_DIR), desc="Извлечение эмбеддингов"):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    path = os.path.join(UNKNOWN_DIR, filename)
    image = Image.open(path)
    if image.mode in ('P', 'RGBA'):
        bg = Image.new("RGB", image.size, (255, 255, 255))
        image = image.convert("RGBA")
        bg.paste(image, mask=image.split()[3])
        image = bg
    else:
        image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        if probs.max().item() < THRESHOLD:
            emb = embedding_model(tensor).squeeze().cpu().numpy()
            embeddings.append(emb)
            paths.append(path)

# --- Кластеризация ---
if embeddings:
    embeddings = np.array(embeddings)
    labels = DBSCAN(eps=3.0, min_samples=3).fit_predict(embeddings)

    for label, path in zip(labels, paths):
        cluster_name = f"cluster_{label}" if label != -1 else "noise"
        cluster_folder = os.path.join(CLUSTER_DIR, cluster_name)
        os.makedirs(cluster_folder, exist_ok=True)
        shutil.copy(path, os.path.join(cluster_folder, os.path.basename(path)))

    # --- Перемещение в новые классы ---
    for label in set(labels):
        if label == -1: continue
        cluster_folder = os.path.join(CLUSTER_DIR, f"cluster_{label}")
        new_class_folder = os.path.join(NEW_CLASS_DIR, f"new_class_{label}")
        os.makedirs(new_class_folder, exist_ok=True)
        for img in os.listdir(cluster_folder):
            shutil.move(os.path.join(cluster_folder, img),
                        os.path.join(new_class_folder, img))
    print(f"Новые классы сохранены в {NEW_CLASS_DIR}")

    # --- Визуализация ---
    tsne = TSNE(n_components=2, random_state=0).fit_transform(embeddings)
    umap_results = umap.UMAP(n_components=2, random_state=0).fit_transform(embeddings)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='Spectral')
    plt.title("t-SNE")
    plt.subplot(1, 2, 2)
    plt.scatter(umap_results[:, 0], umap_results[:, 1], c=labels, cmap='Spectral')
    plt.title("UMAP")
    plt.show()

else:
    print("Нет изображений для кластеризации.")
