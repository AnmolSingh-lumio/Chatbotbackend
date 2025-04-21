import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    raise ValueError("DATABASE_URL not found in environment")

# Add SSL mode if not present for Render
if "render.com" in DATABASE_URL and "sslmode" not in DATABASE_URL:
    DATABASE_URL += "?sslmode=require"

# Create engine with proper SSL settings
engine_args = {}
if "render.com" in DATABASE_URL:
    engine_args.update({
        "connect_args": {
            "sslmode": "require"
        }
    })

# Create engine
print(f"Connecting to database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'masked-url'}")
engine = create_engine(DATABASE_URL, **engine_args)

# Enable pgvector extension
try:
    with engine.connect() as conn:
        print("Enabling pgvector extension...")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        print("pgvector extension enabled successfully")
except Exception as e:
    print(f"Error enabling pgvector extension: {str(e)}")
    # Continue execution, as the error might be that the extension is already enabled
    print("Continuing deployment process...")
