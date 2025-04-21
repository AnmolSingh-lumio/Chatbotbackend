"""
Database connection and models for the chatbot application.
"""
import os
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey, func, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    logger.warning("DATABASE_URL not found in environment. Using default local PostgreSQL instance.")
    DATABASE_URL = "postgresql://postgres:password@localhost:5432/chatbot"
else:
    # Handle Render's SSL requirement
    if "sslmode" not in DATABASE_URL and "render.com" in DATABASE_URL:
        DATABASE_URL += "?sslmode=require"

# Convert synchronous URL to async
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Create engine with proper SSL settings
engine_args = {"pool_pre_ping": True}
if "render.com" in DATABASE_URL:
    engine_args.update({
        "connect_args": {
            "sslmode": "require"
        }
    })

# Create engine and session
engine = create_engine(DATABASE_URL, **engine_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create async engine and session with same SSL settings
async_engine_args = {"pool_pre_ping": True}
if "render.com" in ASYNC_DATABASE_URL:
    async_engine_args.update({
        "ssl": True
    })

async_engine = create_async_engine(ASYNC_DATABASE_URL, **async_engine_args)
AsyncSessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=async_engine, 
    class_=AsyncSession
)

# Create base class for models
Base = declarative_base()

# Define models
class Document(Base):
    """Document model representing uploaded markdown files."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    content = Column(Text)
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """Document chunk model for vector search."""
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index = Column(Integer)
    content = Column(Text)
    embedding = Column(ARRAY(Float))  # Store the embedding vector
    chunk_metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    document = relationship("Document", back_populates="chunks")


def get_db():
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db():
    """Get an async database session."""
    async with AsyncSessionLocal() as session:
        yield session


def init_db():
    """Initialize the database by creating tables."""
    try:
        # Create pgvector extension if not exists
        conn = engine.connect()
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        
        # Create the langchain_pg_collection table if it doesn't exist
        try:
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS langchain_pg_collection (
                name TEXT PRIMARY KEY,
                cmetadata JSONB
            );
            """))
            
            # Create the langchain_pg_embedding table if it doesn't exist
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                uuid UUID PRIMARY KEY,
                collection_name TEXT REFERENCES langchain_pg_collection(name) ON DELETE CASCADE,
                embedding vector(1536),
                document TEXT,
                cmetadata JSONB,
                custom_id TEXT
            );
            """))
            
            conn.commit()
        except Exception as e:
            logger.warning(f"Could not create langchain tables: {str(e)}")
        
        conn.close()
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise