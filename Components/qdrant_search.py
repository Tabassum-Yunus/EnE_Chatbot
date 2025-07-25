from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from .qdrant_store import create_collection_if_not_exists, initialize_qdrant_client
from dotenv import load_dotenv
import os
from nicegui import ui

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
QDRANT_URL = os.getenv('QDRANT_URL')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')


def initialize_embeddings():
    """Initialize and return OpenAI embeddings"""
    if not OPENAI_API_KEY or not EMBEDDING_MODEL:
        raise ValueError("OPENAI_API_KEY or EMBEDDING_MODEL not set in environment variables")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
    return embeddings

def search_similar_question(question, similarity_threshold=0.8, collection_name=COLLECTION_NAME):
    """Search for similar questions in Qdrant vector store"""
    try:
        client = initialize_qdrant_client()
        collection_name = create_collection_if_not_exists(client)
        embeddings = initialize_embeddings()
        
        # Generate vector for the question
        query_vector = embeddings.embed_query(question)
        
        # Search for similar vectors
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=1,
            score_threshold=similarity_threshold
        )
        
        if search_result:
            return {
                "payload": search_result[0].payload,
                "point_id": search_result[0].id  # get the matched point ID
            }
        return None
    except Exception as e:
        raise Exception(f"Failed to search in Qdrant: {str(e)}")


def update_timestamp(point_id, timestamp, collection_name=COLLECTION_NAME):
    """Update timestamp of an existing point in Qdrant"""
    try:
        client = initialize_qdrant_client()
        client.set_payload(
            collection_name=collection_name,
            payload={"timestamp": timestamp},
            points=[point_id]
        )
    except Exception as e:
        raise Exception(f"Failed to update timestamp in Qdrant: {str(e)}")
