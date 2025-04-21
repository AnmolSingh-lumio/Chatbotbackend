# Contract Chatbot Backend

This is the backend for a Contract Chatbot that allows users to upload and analyze legal documents and contracts using Natural Language Processing (NLP) and Retrieval Augmented Generation (RAG).

## Architecture

The Contract Chatbot uses a modern RAG (Retrieval Augmented Generation) architecture with the following components:

### Core Components

1. **Document Chunking**: Documents are split into smaller chunks with configurable size (5000 chars) and overlap (100 chars)
2. **Vector Embeddings**: OpenAI embeddings are used to convert text chunks to vector representations
3. **Vector Database**: PostgreSQL with pgvector extension stores document embeddings for semantic search
4. **RAG with Function Calling**: OpenAI's function calling capabilities for structured retrieval and response generation

### Technical Stack

- **Backend**: FastAPI for high-performance REST API
- **Database**: PostgreSQL with pgvector extension
- **NLP**: OpenAI's API for embeddings and text generation
- **Deployment**: Render.com for hosting

## Features

- Upload, store, and manage markdown documents
- Split documents into semantic chunks with appropriate overlap
- Generate vector embeddings for efficient semantic search
- Query documents using natural language
- Retrieve relevant document chunks based on semantic similarity
- Generate accurate answers using OpenAI with function calling
- Multi-document support for comparing contracts

## API Endpoints

- `POST /api/qa`: Ask questions about a document
- `GET /api/files`: List all uploaded documents
- `POST /api/upload`: Upload a new document
- `GET /api/file/{filename}`: Get content of a specific document
- `DELETE /api/files/{filename}`: Delete a document
- `GET /api/status`: Check API status and readiness
- `GET /api/health`: Health check endpoint

## Setup and Deployment

### Environment Variables

The following environment variables need to be set:

```
# API Keys
OPENAI_API_KEY=your-openai-api-key

# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/chatbot
VECTOR_DIMENSION=1536  # Dimension of OpenAI embeddings

# Application Settings
CHUNK_SIZE=5000
CHUNK_OVERLAP=100
MAX_DOCUMENTS=100
MODEL_NAME=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
```

### Local Development

1. Clone the repository
2. Install PostgreSQL and create a database
3. Install pgvector extension: `CREATE EXTENSION vector;`
4. Install dependencies: `pip install -r requirements.txt`
5. Set environment variables or create a `.env` file
6. Run the application: `python -m uvicorn main:app --reload`

### Deployment

The application is configured for deployment on Render.com with the included `render.yaml` file.

## Implementation Notes

### Chunking Strategy

Documents are chunked using a sophisticated strategy:
1. First tried to split by markdown headers to preserve semantic structure
2. If sections are too large, split by paragraphs
3. If paragraphs are too large, split by sentences
4. Always maintain context with proper overlap between chunks

### Vector Search

Vector similarity search is performed using PostgreSQL's pgvector with cosine similarity.

### OpenAI Function Calling

The system uses OpenAI's function calling to implement a more structured RAG pattern:
1. The model is given tools to query the document database
2. It decides what information to retrieve based on the question
3. It can retrieve multiple chunks to build a comprehensive answer
4. It can request document metadata when needed

## Future Improvements

- Add support for more document formats (PDF, DOCX)
- Implement document comparison features
- Add user authentication and document access control
- Improve chunking with hierarchical embeddings
- Add support for fine-tuned embeddings for legal documents 