# Inicializace databáze (v paměti pro bezproblémový chod na Streamlit Cloud)
@st.cache_resource
def get_qdrant_client():
    client = QdrantClient(":memory:")
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE)
        )
    return client
