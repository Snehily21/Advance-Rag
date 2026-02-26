from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient, models
import os
from dotenv import load_dotenv

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# ---------------------------
# Load PDF
# ---------------------------
pdf_path = Path(__file__).parent / "pypdf.pdf"
docs = PyPDFLoader(str(pdf_path)).load()

# ---------------------------
# Chunking
# ---------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# ---------------------------
# Embeddings
# ---------------------------
embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=OPENAI_API_KEY
)

vectors = embedder.embed_documents(
    [c.page_content for c in chunks]
)

# ---------------------------
# Qdrant Client
# ---------------------------
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

collection_name = "advance_langchain"

# Create collection only if not exists
if collection_name not in [c.name for c in client.get_collections().collections]:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=len(vectors[0]),
            distance=models.Distance.COSINE
        )
    )

# ---------------------------
# Upload Points
# ---------------------------
client.upload_points(
    collection_name=collection_name,
    points=[
        models.PointStruct(
            id=i,
            vector=vectors[i],
            payload={
                "page": chunks[i].metadata.get("page"),
                "source": chunks[i].metadata.get("source"),
                "text": chunks[i].page_content
            }
        )
        for i in range(len(vectors))
    ]
)

print("✅ Ingestion completed successfully with metadata")