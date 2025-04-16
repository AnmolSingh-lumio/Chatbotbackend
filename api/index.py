from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main FastAPI application
from main import app

# This file exists to serve as a serverless function entry point for Vercel
# Vercel will use this file as the handler for all API requests

async def handler(request: Request):
    """
    Serverless function handler for Vercel.
    This catches all requests and forwards them to the FastAPI application.
    """
    return await app(request.scope, request._receive, request._send) 