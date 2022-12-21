"""
A Streamlit application to visualize sentence embeddings

Author: Mohit Mayank
Contact: mohitmayank1@gmail.com
"""

## Import
## ----------------
# data
import pandas as pd
# model
from sentence_transformers import SentenceTransformer, util
# viz
import streamlit as st
import plotly.express as px
# DR
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

## Init
## ----------------
# set config
# st.set_page_config(layout="wide", page_title="SentenceViz 🕵")
st.markdown("# SentenceViz")
st.markdown("A Streamlit application to visulize sentence embeddings")

# load the summarization model (cache for faster loading)
@st.cache(allow_output_mutation=True)
def load_similarity_model(model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    return model

@st.cache(allow_output_mutation=True)
def perform_embedding(df, text_col_name):
    embeddings = model.encode(df[text_col_name])
    return embeddings

# gloabl vars
df = None
model = None
embeddings = None

## Design Sidebar
## -----------------
## Data
st.sidebar.markdown("## Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with sentences (we remove NaN)")
if uploaded_file is not None:
    progress = st.empty()
    progress.text("Reading file...")
    df = pd.read_csv(uploaded_file).dropna().reset_index(drop=True)
    progress.text(f"Reading file...Done! Size: {df.shape[0]}")

## Embedding
st.sidebar.markdown("## Embedding")
supported_models = ['all-MiniLM-L6-v2', 'paraphrase-albert-small-v2', 'paraphrase-MiniLM-L3-v2', 'all-distilroberta-v1', 'all-mpnet-base-v2']
selected_model_option = st.sidebar.selectbox("Select Model:", supported_models)
text_col_name = st.sidebar.text_input("Text column to embed")
if len(text_col_name) > 0 and df is not None:
    df[text_col_name] = df[text_col_name].str.wrap(30)
    df[text_col_name] = df[text_col_name].apply(lambda x: x.replace('\n', '<br>'))
    progress = st.empty()
    progress.text("Creating embedding...")
    model = load_similarity_model(selected_model_option)
    embeddings = perform_embedding(df, text_col_name)
    progress.text("Creating embedding...Done!")

## Visualization
st.sidebar.markdown("## Visualization")
dr_algo = st.sidebar.selectbox("Dimensionality Reduction Algorithm", ('PCA', 't-SNE', 'UMAP'))
color_col = st.sidebar.text_input("Color using this col")
if len(color_col.strip()) == 0:
    color_col = None

if st.sidebar.button('Plot!'):
    # get the embeddings and perform DR
    if dr_algo == 'PCA':
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
    elif dr_algo == 't-SNE':
        tsne = TSNE(n_components=2)
        reduced_embeddings = tsne.fit_transform(embeddings)
    elif dr_algo == 'UMAP':
        reducer = umap.UMAP(random_state=42)
        reducer.fit(embeddings)
        reduced_embeddings = reducer.transform(embeddings)

    
    # modify the df
    # df['complete_embeddings'] = embeddings
    df['viz_embeddings_x'] = reduced_embeddings[:, 0]
    df['viz_embeddings_y'] = reduced_embeddings[:, 1]

    # plot the data
    fig = px.scatter(df, x='viz_embeddings_x', y='viz_embeddings_y', 
            title=f'"{dr_algo}" on {df.shape[0]} "{selected_model_option}" embeddings', 
            color=color_col, hover_data=[text_col_name])
    fig.update_layout(yaxis={'visible': False, 'showticklabels': False})
    fig.update_layout(xaxis={'visible': False, 'showticklabels': False})
    fig.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1,color='DarkSlateGrey')),selector=dict(mode='markers'))
    st.plotly_chart(fig, use_container_width=True)