# EnE Chatbot

## Overview

EnE Chatbot is a Retrieval-Augmented Generation (RAG) chatbot designed to provide precise answers by retrieving information directly from the Engineering and Environmental Solutions website. It helps users quickly find relevant information without browsing the entire site manually.

## Features

- **RAG-based** chatbot leveraging website content as knowledge base  
- Provides accurate, context-aware responses  
- Built specifically for Engineering and Environmental Solutions domain  
- Easy integration with the company’s website  

---

## Getting Started

Follow the steps below to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Tabassum-Yunus/EnE_Chatbot.git
cd EnE_Chatbot

### 2. Create and Activate a Virtual Environment

**On Windows:**

```bash
python -m venv py_env
.\py_env\Scripts\activate
```

**On Unix or MacOS:**

```bash
python3 -m venv py_env
source py_env/bin/activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Create a `.env` file in the root directory with the following content:

```env
OPENAI_API_KEY=<Your OpenAI API key>

EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
CHAT_MODEL=gpt-3.5-turbo

QDRANT_URL=<Your Qdrant host URL>
QDRANT_API_KEY=<Your Qdrant API key>
COLLECTION_NAME=chat_history
```

### 5. Run the Project

```bash
python main.py
```

---

## How to Get API Keys

### OpenAI API Key

1. Visit the [OpenAI API Keys page](https://platform.openai.com/settings/organization/api-keys)
2. Click **Create new secret key** and copy the key into the `.env` file under `OPENAI_API_KEY`.

### Qdrant API Key and URL

1. Go to [Qdrant Cloud](https://cloud.qdrant.io/login) and log in or sign up.
2. In the sidebar, click **Clusters**.
3. Create a new cluster (choose **Free Cluster**).
4. After creation, a popup will display your **Qdrant API Key** – use this for `QDRANT_API_KEY`.
5. In the **Use the API** section, copy the **Endpoint** and use it as `QDRANT_URL`.

   * Make sure to remove any port (like `:6333`) if present.
   * It should look like:

     ```
     https://7bf6-----3edd.cloud.qdrant.io
     ```

---

## Configuration

| Variable               | Description                           |
| ---------------------- | ------------------------------------- |
| `OPENAI_API_KEY`       | Your OpenAI API key                   |
| `EMBEDDING_MODEL`      | Model used for embeddings             |
| `EMBEDDING_DIMENSIONS` | Dimensionality of the embedding model |
| `CHAT_MODEL`           | Chat model used for responses         |
| `QDRANT_URL`           | Qdrant host endpoint                  |
| `QDRANT_API_KEY`       | Qdrant API Key                        |
| `COLLECTION_NAME`      | Name of the Qdrant collection         |

---

## Technologies Used

* [OpenAI GPT-3.5 Turbo](https://platform.openai.com/docs/models/gpt-3-5)
* [Qdrant Vector DB](https://qdrant.tech/)
* Python 3.8+
