import os
import ast
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# streamlit page setup
st.set_page_config(page_title="Glaze Search", layout="wide")
st.title("Glaze Search Engine")
st.caption("Find pottery glazes that match your description")

# env vars + openai api
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#unified index
@st.cache_data
def load_index():
    df = pd.read_csv("data/processed/glaze_master_index.csv")
    # Convert embeddings stored as strings into numeric lists
    if "embedding_text" in df.columns:
        df["embedding_text"] = df["embedding_text"].apply(ast.literal_eval)
    return df

df = load_index()
st.success(f"Loaded {len(df)} glaze entries with captions, recipes, and embeddings.")

#helper funcs
def embed_text(query: str):
    """Generate a text embedding for the search query."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#streamlit sidebar info
with st.sidebar:
    st.header("About this App")
    st.write("""
    This prototype uses OpenAI embeddings and CLIP image features to match text queries to real glaze photos.  
    Each result shows:
    - The generated **visual caption**
    - The actual **glaze recipe**
    - A **similarity score**
    """)
    st.write("Built by Stef! 💖")

#search box
query = st.text_input(
    "Describe the glaze you're looking for:",
    placeholder="e.g. 'matte white with brown speckles'",
)

#perform search
if query:
    with st.spinner("Searching for matching glazes..."):
        query_vec = embed_text(query)
        df["similarity"] = df["embedding_text"].apply(
            lambda x: cosine_similarity(x, query_vec)
        )
        top_matches = df.sort_values("similarity", ascending=False).head(5)

    st.divider()
    st.subheader("Top Matches")

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
            st.caption(f"Glaze recipe: {row['description']}")
            st.progress(float(row["similarity"]))
