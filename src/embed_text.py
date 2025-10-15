import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

#load env variables (API key)
load_dotenv()
client = OpenAI()

#meta data load
metadata_path = "data/raw/metadata.csv"
df = pd.read_csv(metadata_path)

#generate embeddings
print("Generating text embeddings...")

embeddings = []
for desc in tqdm(df["description"], desc="Embedding glaze descriptions"):
    response = client.embeddings.create(
        input=desc,
        model="text-embedding-3-small"  # cheaper + faster
    )
    embeddings.append(response.data[0].embedding)

df["embedding"] = embeddings

#save output!
output_path = "data/processed/glaze_text_embeddings.csv"
os.makedirs("data/processed", exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Embeddings saved to {output_path}")
