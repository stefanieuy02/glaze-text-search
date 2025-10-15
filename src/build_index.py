import pandas as pd
import os

#paths
text_emb_path = "data/processed/glaze_text_embeddings.csv"
img_emb_path = "data/processed/glaze_image_embeddings.csv"
caption_path = "data/processed/glaze_captions.csv"
metadata_path = "data/raw/metadata.csv"

#output
output_path = "data/processed/glaze_master_index.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# dataset liads
meta_df = pd.read_csv(metadata_path)
cap_df = pd.read_csv(caption_path)
text_df = pd.read_csv(text_emb_path)
img_df = pd.read_csv(img_emb_path)

#drop dup column from metadata.csv
if "description" in text_df.columns:
    text_df = text_df.drop(columns=["description"])

#merge
merged = (
    meta_df
    .merge(cap_df, on="filename", how="left")
    .merge(text_df, on="filename", how="left")
    .merge(img_df, on="filename", how="left", suffixes=("_text", "_img"))
)

#clean up cols
if "description_x" in merged.columns and "description_y" in merged.columns:
    merged.drop(["description_y"], axis=1, inplace=True)
    merged.rename(columns={"description_x": "description"}, inplace=True)

#save merge file
merged.to_csv(output_path, index=False)
print(f"Combined index saved to {output_path}")
print(f"Columns: {merged.columns.tolist()}")