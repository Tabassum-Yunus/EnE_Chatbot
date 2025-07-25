import os
import uuid
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
QDRANT_URL = os.getenv('QDRANT_URL')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
EMBEDDING_DIMENSIONS = os.getenv('EMBEDDING_DIMENSIONS')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

def initialize_qdrant_client():
    """Initialize and return Qdrant client"""
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        return client
    except Exception as e:
        raise Exception(f"Failed to initialize Qdrant client: {str(e)}")

def initialize_embeddings():
    """Initialize and return OpenAI embeddings"""
    if not OPENAI_API_KEY or not EMBEDDING_MODEL:
        raise ValueError("OPENAI_API_KEY or EMBEDDING_MODEL not set in environment variables")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
    return embeddings

def create_collection_if_not_exists(client, collection_name=COLLECTION_NAME):
    
    """Create Qdrant collection if it doesn't exist"""
    try:
        
        collections = client.get_collections()
        collection_exists = any(c.name == collection_name for c in collections.collections)
        
        if not collection_exists:
            print(f"Creating collection '{collection_name}' with dimension {EMBEDDING_DIMENSIONS}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSIONS,  
                    distance=Distance.COSINE
                )
            )

        return collection_name
    except Exception as e:
        raise Exception(f"Failed to create Qdrant collection: {str(e)}")

def store_question_response(question, response, timestamp):
    """Store question and response in Qdrant vector store"""
    try:
        print('question: ', question)
        client = initialize_qdrant_client()
        embeddings = initialize_embeddings()
        collection_name = create_collection_if_not_exists(client)
        
        # Generate vector for the question
        query_vector = embeddings.embed_query(question)
        
        # Create point with question, response, and timestamp as metadata
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=query_vector,
            payload={
                "question": question,
                "response": response,
                "timestamp": timestamp
            }
        )
        
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
    except Exception as e:
        raise Exception(f"Failed to store in Qdrant: {str(e)}")