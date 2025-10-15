import os
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import CLIPProcessor, CLIPModel

model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)

input_dir = "data/raw/images"
output_path = "data/processed/glaze_image_embeddings.csv"

os.makedirs(os.path.dirname(output_path), exist_ok=True)
results = []

for fname in tqdm(os.listdir(input_dir)):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(os.path.join(input_dir, fname)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            emb = model.get_image_features(**inputs).squeeze().tolist()
        results.append({"filename": fname, "embedding": emb})

pd.DataFrame(results).to_csv(output_path, index=False)
print(f"Saved image embeddings to {output_path}")