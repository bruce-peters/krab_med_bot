"""
Krab Med Bot - Main FastAPI Application
AI-powered medication management system for elderly users
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import logging

from server.config import settings
from server.controllers.hardware_interface import hardware_interface
from server.controllers.mock_hardware import mock_hardware_interface
from server.routes import hardware, medication, health_data, ai_conversation, voice
from server.utils.json_handler import initialize_data_file

# Setup logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup
    logger.info("🚀 Starting Krab Med Bot API...")
    logger.info("📦 Initializing components...")
    
    # Initialize data files
    logger.info("📁 Initializing data files...")
    await initialize_data_file("data/medication_schedule.json", {
        "user_id": "user_001",
        "medications": []
    })
    await initialize_data_file("data/health_logs.json", [])
    await initialize_data_file("data/user_interactions.json", [])
    
    # Initialize hardware interface based on mode
    if settings.hardware_mode == "mock":
        logger.info("🔧 Running in MOCK hardware mode")
        app.state.hardware = mock_hardware_interface
        await mock_hardware_interface.initialize()
    else:
        logger.info("🔧 Running in PRODUCTION hardware mode")
        app.state.hardware = hardware_interface
        await hardware_interface.initialize()
        
        # Test connection to actual hardware
        try:
            connected = await hardware_interface.test_connection()
            if connected:
                logger.info("✅ Hardware controller connected")
            else:
                logger.warning("⚠️  Hardware controller not reachable")
        except Exception as e:
            logger.error(f"❌ Failed to connect to hardware controller: {e}")
    
    logger.info("✅ Startup complete!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Krab Med Bot API...")
    
    # Cleanup hardware
    if settings.hardware_mode == "mock":
        await mock_hardware_interface.close()
    else:
        await hardware_interface.close()
    
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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins if settings.cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(hardware.router)
app.include_router(medication.router)
app.include_router(health_data.router)
app.include_router(ai_conversation.router)  # NEW
app.include_router(voice.router)            # NEW


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "🤖 Krab Med Bot API",
        "version": "1.0.0",
        "status": "running",
        "hardware_mode": settings.hardware_mode,
        "docs": "/docs",
        "health": "/health",
        "ai_enabled": True,
        "ai_provider": settings.ai_provider,
        "voice_enabled": settings.enable_voice_interaction
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check hardware connection
    hardware_status = "unknown"
    try:
        connected = await app.state.hardware.test_connection()
        hardware_status = "connected" if connected else "disconnected"
    except Exception:
        hardware_status = "error"
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "Krab Med Bot API",
        "hardware_mode": settings.hardware_mode,
        "hardware_status": hardware_status,
        "ai_provider": settings.ai_provider
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower()
    )
