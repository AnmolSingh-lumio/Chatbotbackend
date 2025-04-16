# Chatbot Backend

Backend API for the markdown document chatbot application.

## Deployment on Render

### Manual Deployment

1. Create a new Web Service in Render
2. Connect your GitHub repository
3. Use the following settings:
   - **Name**: chatbot-backend (or your preferred name)
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT`

4. Add the following environment variables:
   - `PYTHONPATH`: `.`
   - `ENVIRONMENT`: `production`
   - `GEMINI_API_KEY`: Your Google Gemini API key
   - `SUPABASE_URL`: Your Supabase project URL
   - `SUPABASE_KEY`: Your Supabase project API key

### Deployment with Blueprint

Alternatively, you can use Render's Blueprint for automatic deployment:

1. Push this repository to GitHub/GitLab
2. In Render Dashboard, create a "New Blueprint"
3. Connect your repository
4. Render will use the `render.yaml` configuration file

## Local Development

1. Create a `.env` file with your environment variables:
```
GEMINI_API_KEY=your_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the development server:
```
python main.py
```

The API will be available at http://localhost:8000.

## API Documentation

API documentation is available at `/api/docs` or `/api/redoc` endpoints.

## Frontend Integration

Update your frontend environment variables to point to your Render deployment URL:

```
NEXT_PUBLIC_API_URL=https://your-render-app-name.onrender.com
```

## Features

- Q&A on markdown documents using RAG (Retrieval-Augmented Generation)
- Question embedding and template matching for better answers
- File upload, management, and retrieval endpoints
- Uses Gemini API for embeddings and answering questions

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the `.env.example` file to `.env` and update with your Gemini API key:
   ```bash
   cp .env.example .env
   ```
5. Run the application:
   ```bash
   uvicorn main:app --reload
   ```

## API Endpoints

- **GET /api/health**: Health check endpoint
- **POST /api/qa**: Ask a question about a markdown document
- **POST /api/upload**: Upload a markdown document
- **GET /api/files**: List all uploaded markdown files
- **GET /api/file/{filename}**: Get the content of a specific markdown file

## Usage

1. Upload a markdown document using the `/api/upload` endpoint
2. Get the content of the uploaded document using the `/api/file/{filename}` endpoint
3. Ask questions about the document using the `/api/qa` endpoint with the document content and your question

## Example

```json
// Request
POST /api/qa
{
  "markdown_content": "# Sample Document\n\nThis is a sample markdown document.\n\n## Features\n\n- Feature 1\n- Feature 2\n- Feature 3",
  "question": "What features are listed in the document?"
}

// Response
{
  "success": true,
  "message": "Question answered successfully",
  "data": {
    "answer": "The document lists three features: Feature 1, Feature 2, and Feature 3.",
    "confidence": 0.95,
    "source_documents": []
  }
}
```

## License

MIT 