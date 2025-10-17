import pandas as pd
import os

#paths
text_emb_path = "data/processed/glaze_text_embeddings.csv"
img_emb_path = "data/processed/glaze_image_embeddings.csv"
caption_path = "data/processed/glaze_captions.csv"

#output
output_path = "data/processed/glaze_master_index.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# dataset liads
cap_df = pd.read_csv(caption_path)
text_df = pd.read_csv(text_emb_path)
img_df = pd.read_csv(img_emb_path)


#merge
merged = (
    cap_df
    .merge(text_df, on="filename", how="left")
    .merge(img_df, on="filename", how="left")
)


#save merge file
merged.to_csv(output_path, index=False)
print(f"Combined index saved to {output_path}")
print(f"Columns: {merged.columns.tolist()}")