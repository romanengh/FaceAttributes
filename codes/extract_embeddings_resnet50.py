# scripts/extract_embeddings_resnet50.py

import torch
from torch import nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
import os
import time
from PIL import Image

# 1. Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(PROJECT_ROOT, "img_align_celeba")

batch_size = 64

# 2. Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 3. Dataset
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [
            f for f in sorted(os.listdir(root_dir))
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.files[idx]

dataset = SimpleDataset(IMG_DIR, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

print(f"Number of images found: {len(dataset)}")

# 4. Pre-trained ResNet50
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Remove the final classification layer
model = nn.Sequential(*list(model.children())[:-1])

model = model.to(device)
model.eval()


# 5. Embedding extraction
embeddings = []
image_names = []

start_time = time.time()
total_batches = len(dataloader)

print("\nExtracting ResNet50 embeddings...\n")

with torch.no_grad():
    for i, (imgs, names) in enumerate(dataloader):
        imgs = imgs.to(device)

        feats = model(imgs)              # (B, 2048, 1, 1)
        feats = feats.view(feats.size(0), -1)  # (B, 2048)

        embeddings.append(feats.cpu())
        image_names.extend(names)

        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            avg = elapsed / (i + 1)
            remaining = avg * (total_batches - i - 1)

            print(
                f"[Batch {i}/{total_batches}] "
                f"Elapsed: {elapsed/60:.1f} min | "
                f"Remaining: {remaining/60:.1f} min"
            )

# 6. Save embeddings

embeddings = torch.cat(embeddings)

output_path = os.path.join(PROJECT_ROOT, "embeddings_resnet50.pt")

torch.save(
    {
        "embeddings": embeddings,
        "image_names": image_names
    },
    output_path
)

total_time = (time.time() - start_time) / 60

print(f"\nFinished in {total_time:.1f} minutes")
print(f"Embeddings shape: {embeddings.shape}")  # should be (N, 2048)
print(f"Saved to: {output_path}")