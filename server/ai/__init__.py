"""
AI Integration Module for Krab Med Bot
Provides LLM, conversation, symptom analysis, and speech capabilities
"""

from .llm_client import llm_client
from .conversation import conversation_manager
from .symptom_analyzer import symptom_analyzer
from .recommendation import recommendation_engine
from .speech import speech_service
from .prompts import (
    get_system_prompt,
    get_symptom_extraction_prompt,
    get_recommendation_prompt,
    get_greeting_prompt,
    get_follow_up_prompt,
    get_closing_prompt
)

__all__ = [
    "llm_client",
    "conversation_manager",
    "symptom_analyzer",
    "recommendation_engine",
    "speech_service",
    "get_system_prompt",
    "get_symptom_extraction_prompt",
    "get_recommendation_prompt",
    "get_greeting_prompt",
    "get_follow_up_prompt",
    "get_closing_prompt"
]
