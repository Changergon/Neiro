import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def safe_pil_loader(path):
    try:
        image = Image.open(path).convert("RGB")
        return image
    except Exception as e:
        print(f"Ошибка при загрузке изображения {path}: {e}")
        return None

def get_embeddings_from_batch(model, embedder, inputs, batch, threshold):
    batch_embeddings = []
    batch_paths = []
    with torch.no_grad():
        outputs = embedder(inputs)
        for i, emb in enumerate(outputs.cpu().numpy()):
            if np.max(emb) >= threshold:
                batch_embeddings.append(emb)
                batch_paths.append(batch[i][1])
    return batch_embeddings, batch_paths

def extract_embeddings(model, embedder, transform, paths, device, batch_size, threshold):
    embeddings = []
    valid_paths = []
    batch = []
    for path in tqdm(paths, desc="Извлечение эмбеддингов"):
        image = safe_pil_loader(path)
        if image is None:
            continue
        tensor = transform(image).unsqueeze(0)
        batch.append((tensor, path))

        if len(batch) == batch_size:
            inputs = torch.cat([item[0] for item in batch]).to(device)
            batch_embeddings, batch_paths = get_embeddings_from_batch(model, embedder, inputs, batch, threshold)
            embeddings.extend(batch_embeddings)
            valid_paths.extend(batch_paths)
            batch = []

    if batch:
        inputs = torch.cat([item[0] for item in batch]).to(device)
        batch_embeddings, batch_paths = get_embeddings_from_batch(model, embedder, inputs, batch, threshold)
        embeddings.extend(batch_embeddings)
        valid_paths.extend(batch_paths)

    return np.array(embeddings), valid_paths
