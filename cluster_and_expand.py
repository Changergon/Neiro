import os
import pickle
import shutil

import torch
import torch.nn as nn
import umap
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from torchvision import models

from utils import extract_embeddings, get_transforms


def run_clustering(unknown_dir, cluster_dir, new_class_dir, model_path, num_classes, threshold, batch_size, eps, min_samples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_transforms()

    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    embedder = nn.Sequential(*list(model.children())[:-1]).to(device).eval()

    image_paths = [os.path.join(unknown_dir, f) for f in os.listdir(unknown_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    embeddings, valid_paths = extract_embeddings(model, embedder, transform, image_paths, device, batch_size, threshold)

    if len(embeddings) == 0:
        print("Нет изображений для кластеризации.")
        return

    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(embeddings)

    os.makedirs(cluster_dir, exist_ok=True)
    for label, path in zip(labels, valid_paths):
        cluster_name = f"cluster_{label}" if label != -1 else "noise"
        cluster_folder = os.path.join(cluster_dir, cluster_name)
        os.makedirs(cluster_folder, exist_ok=True)
        shutil.copy(path, os.path.join(cluster_folder, os.path.basename(path)))

    for label in set(labels):
        if label == -1: continue
        cluster_folder = os.path.join(cluster_dir, f"cluster_{label}")
        new_class_folder = os.path.join(new_class_dir, f"new_class_{label}")
        os.makedirs(new_class_folder, exist_ok=True)
        for img in os.listdir(cluster_folder):
            shutil.move(os.path.join(cluster_folder, img),
                        os.path.join(new_class_folder, img))

    print(f"Новые классы сохранены в {new_class_dir}")

    with open("embeddings.pkl", "wb") as f:
        pickle.dump({"embeddings": embeddings, "labels": labels}, f)

    tsne = TSNE(n_components=2, random_state=0).fit_transform(embeddings)
    umap_results = umap.UMAP(n_components=2, random_state=0).fit_transform(embeddings)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='Spectral')
    plt.title("t-SNE")
    plt.subplot(1, 2, 2)
    plt.scatter(umap_results[:, 0], umap_results[:, 1], c=labels, cmap='Spectral')
    plt.title("UMAP")
    plt.tight_layout()
    plt.savefig("clusters_visualization.png")
    print("Визуализация сохранена в clusters_visualization.png")
