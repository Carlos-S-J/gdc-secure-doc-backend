
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
from typing import List

import sqlalchemy
from sqlalchemy import text
from sklearn.decomposition import PCA
from google.cloud.sql.connector import Connector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector

app = FastAPI()

# --- Database and Embedding Configuration ---

# GCP Project and DB details from your request
PROJECT_ID = "mods-labs25lon-107"
REGION = "europe-west2"
INSTANCE_NAME = "postgres-db-ts"
DB_USER = "postgres"
DB_PASS = "London2025$"
DB_NAME = "doc_vectors"

# Cloud SQL instance connection name
INSTANCE_CONNECTION_NAME = f"{PROJECT_ID}:{REGION}:{INSTANCE_NAME}"
COLLECTION_NAME = "doc_embeddings"

# Initialize Vertex AI Embeddings
embeddings = VertexAIEmbeddings(
    model_name="gemini-embedding-001"
)

# Initialize Cloud SQL Python Connector
connector = Connector()

def getconn() -> sqlalchemy.engine.base.Connection:
    """Creates a database connection pool for Cloud SQL."""
    conn = connector.connect(
        INSTANCE_CONNECTION_NAME,
        "pg8000",
        user=DB_USER,
        password=DB_PASS,
        db=DB_NAME,
    )
    return conn

# Create a SQLAlchemy engine
engine = sqlalchemy.create_engine("postgresql+pg8000://", creator=getconn)

# Initialize the PGVector store
db = PGVector(
    connection_string=f"postgresql+pg8000://{DB_USER}:{DB_PASS}@/{DB_NAME}?host={INSTANCE_CONNECTION_NAME}",
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    engine_args={"creator": getconn},
)

@app.get("/")
def hello(name: str = "World"):
    """Return a friendly HTTP greeting."""
    return {
    "message": f"Hello {name}!"
}

@app.post("/upload-pdfs/")
async def upload_and_embed_pdfs(files: List[UploadFile] = File(...)):
    """
    Accepts two PDF files, extracts text, chunks it, creates embeddings,
    and stores them in a GCP PostgreSQL (pgvector) database.
    """
    if len(files) != 2:
        raise HTTPException(status_code=400, detail="Please upload exactly two PDF files.")

    all_docs = []
    for file in files:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not a PDF.")

        # Save the uploaded file temporarily to be read by PyPDFLoader
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        # Load documents from the PDF file
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        all_docs.extend(docs)

        # Clean up the temporary file
        os.remove(temp_file_path)

    if not all_docs:
        raise HTTPException(status_code=400, detail="Could not extract any text from the provided PDFs.")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunked_docs = text_splitter.split_documents(all_docs)

    # Add documents to the vector store
    # This will create embeddings and store them in the database
    db.add_documents(chunked_docs)    

    return {
        "message": f"Successfully processed and embedded {len(files)} PDF files.",
        "total_chunks_added": len(chunked_docs)
    }

@app.get("/embeddings-2d/")
def get_embeddings_2d():
    """
    Retrieves all embeddings, reduces them to 2 dimensions using PCA,
    and returns them with their document source.
    """
    try:
        with engine.connect() as conn:
            # Using COLLECTION_NAME which is 'doc_embeddings'
            # PGVector stores metadata in 'cmetadata' column
            query = text(f'SELECT embedding, `cmetadata` FROM "{COLLECTION_NAME}"')
            # PGVector stores embeddings in 'langchain_pg_embedding' and collection
            # info in 'langchain_pg_collection'. We need to join them to get the
            # embeddings for the correct collection.
            query = text(f"""
                SELECT embedding, `cmetadata` FROM langchain_pg_embedding
                JOIN langchain_pg_collection ON langchain_pg_embedding.collection_id = langchain_pg_collection.uuid
                WHERE langchain_pg_collection.name = '{COLLECTION_NAME}'
            """)
            results = conn.execute(query).fetchall()

        if not results:
            return {"message": "No embeddings found in the database."}

        embeddings_list = [np.array(row[0]) for row in results]
        metadata_list = [row[1] for row in results]

        # Ensure we have enough samples for PCA
        if len(embeddings_list) < 2:
            raise HTTPException(status_code=400, detail="Not enough data points to generate a 2D projection. At least 2 are required.")

        # Reduce dimensionality to 2D using PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(np.array(embeddings_list))

        # Prepare the response
        response_data = [{"source": meta.get("source", "Unknown"), "x": float(point[0]), "y": float(point[1])} for meta, point in zip(metadata_list, embeddings_2d)]

        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
