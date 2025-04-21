import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    raise ValueError("DATABASE_URL not found in environment")

# Create engine
engine = create_engine(DATABASE_URL)

# Enable pgvector extension
with engine.connect() as conn:
    print("Enabling pgvector extension...")
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()
    print("pgvector extension enabled successfully")
