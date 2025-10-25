"""
Configuration Management using Pydantic Settings
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Union, Optional
import json


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
    
    # CORS settings - accepts JSON string or list
    cors_origins: Union[List[str], str] = '["http://localhost:3000", "http://localhost:5173"]'
    
    # AI/LLM Settings
    ai_provider: str = "gemini"  # "openai", "anthropic", "ollama", "gemini", "mock"
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    
    # LLM Model Settings
    llm_model: str = "gpt-4"  # Latest model with better safety handling
    llm_temperature: float = 0.7
    llm_max_tokens: int = 100000
    
    # Speech Settings
    stt_provider: str = "openai"  # "openai", "gemini", "whisper_local", "mock"
    tts_provider: str = "openai"  # "openai", "local", "elevenlabs", "mock"
    tts_voice: str = "echo"  # Voice name (for providers that support it)
    
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Parse cors_origins if it's a JSON string
        if isinstance(self.cors_origins, str):
            try:
                self.cors_origins = json.loads(self.cors_origins)
            except json.JSONDecodeError:
                # Fallback: split by comma
                self.cors_origins = [
                    origin.strip() 
                    for origin in self.cors_origins.split(',')
                ]


# Create global settings instance
settings = Settings()


# Validate settings on import
def validate_settings():
    """Validate critical settings"""
    if settings.ai_provider not in ["mock", "openai", "anthropic", "ollama", "gemini"]:
        raise ValueError(f"Invalid AI provider: {settings.ai_provider}")
    
    if settings.hardware_mode not in ["mock", "production"]:
        raise ValueError(f"Invalid hardware mode: {settings.hardware_mode}")
    
    if settings.ai_provider == "openai" and not settings.openai_api_key:
        print("⚠️  WARNING: OpenAI API key not set, AI features will be limited")
    
    if settings.hardware_mode == "production" and not settings.hardware_controller_url:
        print("⚠️  WARNING: Hardware controller URL not set")


# Run validation
validate_settings()
