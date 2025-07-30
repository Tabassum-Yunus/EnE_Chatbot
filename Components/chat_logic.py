import os
import uuid
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import traceback
from .qdrant_store import store_question_response
from .qdrant_search import search_similar_question, update_timestamp

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
CHAT_MODEL = os.getenv('CHAT_MODEL')

base_path = Path(__file__).resolve().parent.parent
faiss_path = base_path / "faiss_index"

# Cache global instances
_vector_store = None
_llm = None
_hybrid_retriever = None

def initialize_embeddings():
    '''Initialize and return OpenAI embeddings'''
    if not OPENAI_API_KEY or not EMBEDDING_MODEL:
        raise ValueError("OPENAI_API_KEY or EMBEDDING_MODEL not set in environment variables")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
    return embeddings

def initialize_llm():
    '''Initialize and return the language model'''
    global _llm
    if _llm is None:
        if not OPENAI_API_KEY or not CHAT_MODEL:
            raise ValueError("OPENAI_API_KEY or CHAT_MODEL not set in environment variables")
        _llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model_name=CHAT_MODEL,
            temperature=0,
            streaming=True,
        )
    return _llm

def load_faiss_files():
    '''Load existing vector store'''
    global _vector_store
    if _vector_store is None:
        try:
            embeddings = initialize_embeddings()
            if not os.path.exists('faiss_index'):
                raise FileNotFoundError("FAISS index file not found")
            _vector_store = FAISS.load_local(
                faiss_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            raise Exception(f"Failed to load FAISS index: {str(e)}")
    return _vector_store

def setup_retrievers(vector_store):
    '''Setup hybrid retriever'''
    global _hybrid_retriever
    if _hybrid_retriever is None:
        try:
            semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            documents = list(vector_store.docstore._dict.values())
            if not documents:
                raise ValueError("No documents found in FAISS vector store")
            keyword_retriever = BM25Retriever.from_documents(documents)
            keyword_retriever.k = 3
            _hybrid_retriever = EnsembleRetriever(
                retrievers=[semantic_retriever, keyword_retriever],
                weights=[0.7, 0.3]
            )
        except Exception as e:
            raise Exception(f"Error setting up retrievers: {str(e)}")
    return _hybrid_retriever

# Define custom prompt template with formatting instructions
PROMPT_TEMPLATE = """
You are an assistant providing information from official documents for E&E Solutions.
Use the following extracted content to answer the following question as accurately as possible.

Context:
{context}

Question: {question}

Instructions:
- Respond only using the provided context to answer the question as accurately as possible.
- If the context contains relevant information, provide the answer using the exact wording from the context without paraphrasing or summarizing unless explicitly asked. Consolidate any repeated information into a single coherent response based on the preprocessed context, ensuring no duplication.
- If the extracted content is empty or the extracted content does not contain any relevant information to answer the question, just say "I couldn't find this info" and do not generate any follow-up questions.
- If extracting information from a table, provide the relevant table content in the response.
- Ensure the response clearly connects the user's question to the information provided in the context.
- Only if the context contains relevant information, generate 3 follow-up questions that are relevant, specific, and encourage deeper exploration of the topic, guiding the user toward purchasing a product from E&E Solutions. Ensure the questions are concise, natural, and based solely on the provided context or information likely to be in the vector store (index.faiss).
- Format the follow-up questions as a numbered list.
- Our aim is to guide the user toward purchasing a product, so frame questions to highlight product benefits, features, or next steps in the sales process.
- Do not generate follow-up questions if the user has inquired about how to purchase a product. Instead, respond by asking the user to fill out the inquiry form available at the following link: http://localhost:8080/form. 
- Your response should include: "Please <a href="#" onclick="send_event('open_form')">fill out our enquiry form</a>. A sales representative will contact you shortly after submission." and modify the response as per the requirement.
- Do not ask for name, company, or designation directly in chat. Only collect it via the form.
"""

async def get_streaming_response(question):
    '''Get streaming response for the given question'''
    try:
        # Search in Qdrant first
        timestamp = datetime.now().isoformat()
        similar_result = await search_similar_question(question)

        if similar_result:
            point_id = similar_result["point_id"]      # get point id (vector)
            response = similar_result["payload"]["response"]    # get response from payload

            update_timestamp(
                point_id=point_id,
                timestamp=timestamp
            )
            yield response
            print(f"Question '{question}' found in Qdrant vectorstore...")
            return

        print(f"Forwarding question '{question}' to OpenAI...")
        # If no similar question found, proceed with FAISS and LLM
        vector_store = load_faiss_files()
        hybrid_retriever = setup_retrievers(vector_store)
        llm = initialize_llm()

        # Create prompt
        custom_prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        # Create streaming chain
        chain = (
            {"context": hybrid_retriever, "question": RunnablePassthrough()}
            | custom_prompt
            | llm
            | StrOutputParser()
        )

        # Stream response and store in Qdrant
        response = ""
        async for chunk in chain.astream(question):
            response += chunk
            yield chunk
        
        if not (response.startswith("Error:") or response.startswith("I couldn't find this")):
            store_question_response(question, response, timestamp) 

    except Exception as e:
        print(f"Error: Failed to process question in chat_logic- {str(e)}")
        yield f"Error: Failed to process question in chat_logic- {str(e)}"