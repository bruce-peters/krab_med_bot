"""
Krab Med Bot - Main FastAPI Application
AI-powered medication management system for elderly users
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

# Note: We'll create config.py next
# from server.config import settings

# Setup basic logging for now
logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup
    logger.info("🚀 Starting Krab Med Bot API...")
    logger.info("📦 Initializing components...")
    
    # TODO: Initialize hardware interface when ready
    # TODO: Test AI connections when ready
    
    logger.info("✅ Startup complete!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Krab Med Bot API...")
    # TODO: Cleanup tasks
    logger.info("👋 Shutdown complete!")


# Create FastAPI app
app = FastAPI(
    title="Krab Med Bot API",
    description="AI-powered medication management system for elderly users",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS - allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "🤖 Krab Med Bot API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "Krab Med Bot API"
    }


# Simple test endpoint
@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return {
        "message": "API is working!",
        "test": "successful"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )
