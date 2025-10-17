import os
import ast
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# streamlit page setup
st.set_page_config(page_title="Glaze Search", layout="wide")
st.title("Glaze Search Engine")
st.caption("Find pottery glazes that match your description")

# env vars + openai api
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#load clip model for image embeddings
@st.cache_resource
def load_clip():
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)
    return model, processor

clip_model, clip_processor = load_clip()

#load master dataset
@st.cache_data
def load_index():
    df = pd.read_csv("data/processed/glaze_master_index.csv")
    # Convert embeddings stored as strings into numeric lists
    if "embedding_text" in df.columns:
        df["embedding_text"] = df["embedding_text"].apply(ast.literal_eval)
    if "embedding_img" in df.columns:
        df["embedding_img"] = df["embedding_img"].apply(ast.literal_eval)
    return df

df = load_index()
st.success(f"Loaded {len(df)} glaze entries with captions, recipes, and embeddings.")

#helper funcs
#text embeddings
def embed_text(query: str):
    """Generate a text embedding for the search query."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return np.array(response.data[0].embedding)

#image embeddings
def embed_image(uploaded_img):
    """Generate CLIP embedding for uploaded image."""
    image = Image.open(uploaded_img).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        img_emb = clip_model.get_image_features(**inputs).squeeze().numpy()
    return img_emb / np.linalg.norm(img_emb)  # normalize for cosine similarity

#cosine similarity
def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#streamlit sidebar info
with st.sidebar:
    st.header("🔧 Search Options")
    search_mode = st.radio("Choose search type:", ["Text Search", "Image Search"])
    st.markdown("---")
    st.write("Built by Stefanie 💫")


#perform search
if search_mode == "Text Search":
    query = st.text_input(
        "Describe the glaze you're looking for:",
        placeholder="e.g. 'matte white with brown speckles'",
    )

    if query:
        with st.spinner("Searching for matching glazes..."):
            query_vec = embed_text(query)
            df["similarity"] = df["embedding_text"].apply(
                lambda x: cosine_similarity(x, query_vec)
            )
            top_matches = df.sort_values("similarity", ascending=False).head(5)

        st.divider()
        st.subheader("Top Matches (Text Search)")
        for _, row in top_matches.iterrows():
            col1, col2 = st.columns([1, 3])
            with col1:
                image_path = os.path.join("data", "raw", "images", row["filename"])
                if os.path.exists(image_path):
                    st.image(image_path, width=180)
                else:
                    st.text("(no image found)")
            with col2:
                st.markdown(f"**{row['visual_caption']}**")
                if "glaze_description" in row:
                    st.caption(f"Glaze recipe: {row['glaze_description']}")
                st.progress(float(row["similarity"]))

elif search_mode == "Image Search":
    uploaded_img = st.file_uploader("Upload a glaze image", type=["jpg", "jpeg", "png"])
    if uploaded_img is not None:
        st.image(uploaded_img, caption="Uploaded Image", width=200)

        with st.spinner("Finding visually similar glazes..."):
            query_vec = embed_image(uploaded_img)
            df["similarity"] = df["embedding_img"].apply(
                lambda x: cosine_similarity(x, query_vec)
            )
            top_matches = df.sort_values("similarity", ascending=False).head(5)

        st.divider()
        st.subheader("Top Matches (Image Search)")
        for _, row in top_matches.iterrows():
            col1, col2 = st.columns([1, 3])
            with col1:
                image_path = os.path.join("data", "raw", "images", row["filename"])
                if os.path.exists(image_path):
                    st.image(image_path, width=180)
                else:
                    st.text("(no image found)")
            with col2:
                st.markdown(f"**{row['visual_caption']}**")
                if "glaze_description" in row:
                    st.caption(f"Glaze recipe: {row['glaze_description']}")
                st.progress(float(row["similarity"]))