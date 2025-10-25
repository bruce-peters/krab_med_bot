"""
Configuration Management using Pydantic Settings
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Server settings
    app_name: str = "Krab Med Bot API"
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    
    # Hardware settings
    hardware_mode: str = "mock"  # "mock" or "production"
    hardware_controller_url: str = "http://localhost:8080"
    hardware_timeout: int = 5
    
    # Data settings
    data_dir: str = "./data"
    log_level: str = "INFO"
    
    # CORS settings
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # AI/LLM Settings
    ai_provider: str = "mock"  # "openai", "anthropic", "ollama", "mock"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    
    # LLM Model Settings
    llm_model: str = "gpt-4-turbo-preview"  # or "claude-3-opus-20240229"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 500
    
    # Speech Settings
    stt_provider: str = "mock"  # "openai", "whisper_local", "mock"
    tts_provider: str = "mock"  # "openai", "elevenlabs", "local", "mock"
    tts_voice: str = "alloy"  # OpenAI TTS voice
    
    # Conversation Settings
    conversation_context_window: int = 10  # Last N messages to include
    enable_voice_interaction: bool = True
    auto_symptom_extraction: bool = True
    
    # Feature Flags
    enable_ai_features: bool = True
    enable_recommendations: bool = True
    enable_symptom_tracking: bool = True
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore extra fields in .env
    )


# Create global settings instance
settings = Settings()


# Validate settings on import
def validate_settings():
    """Validate critical settings"""
    if settings.ai_provider not in ["mock", "openai", "anthropic", "ollama"]:
        raise ValueError(f"Invalid AI provider: {settings.ai_provider}")
    
    if settings.hardware_mode not in ["mock", "production"]:
        raise ValueError(f"Invalid hardware mode: {settings.hardware_mode}")
    
    if settings.ai_provider == "openai" and not settings.openai_api_key:
        print("⚠️  WARNING: OpenAI API key not set, AI features will be limited")
    
    if settings.hardware_mode == "production" and not settings.hardware_controller_url:
        print("⚠️  WARNING: Hardware controller URL not set")


# Run validation
validate_settings()
