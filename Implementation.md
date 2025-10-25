# AI Medicine Box - Backend Implementation Plan

## Project Overview

A middleware server that bridges hardware components (LEDs) with a frontend interface, managing medication dispensing coordination and health data collection for elderly users. **Features AI-powered conversational interactions, symptom analysis, and personalized health recommendations.** **Note: Servo motor control is handled by separate hardware controller code.**

## Technology Stack

- **Server Framework**: FastAPI (Python) - High performance, async support, automatic API documentation
- **Hardware Interface**: GPIO control for LEDs (RPi.GPIO for Raspberry Pi)
- **External Integration**: HTTP/Serial communication with servo controller
- **AI/LLM Integration**: OpenAI API, Anthropic Claude, or local LLM (Ollama)
- **Speech-to-Text**: OpenAI Whisper API or local Whisper model
- **Text-to-Speech**: OpenAI TTS API, ElevenLabs, or local Piper TTS
- **NLP/Analysis**: spaCy or transformers for symptom extraction
- **Data Format**: JSON for all communications
- **Communication Protocol**: HTTP REST API with WebSockets for real-time updates
- **Async HTTP Client**: httpx for async requests to servo controller and AI APIs
- **Validation**: Pydantic models for request/response validation

## System Architecture

```
[Frontend] <--> [FastAPI Server] <--> [LED Controller]
                       |                      |
                   [AI Engine]         [External Servo Controller]
                       |
        [OpenAI/Claude/Ollama API]
        [Whisper STT / TTS Engine]
                       |
                   [Data Storage]
                   [Conversation History]
```

## Detailed Implementation Steps

### Phase 1: Project Setup

#### Step 1.1: Initialize Project Structure

```
krab_med_bot/
├── server/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration and settings
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── medication.py
│   │   ├── hardware.py
│   │   ├── health_data.py
│   │   ├── ai_conversation.py  # NEW: AI conversation endpoints
│   │   └── voice.py            # NEW: Voice interaction endpoints
│   ├── controllers/
│   │   ├── __init__.py
│   │   ├── led_controller.py
│   │   └── servo_interface.py
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── llm_client.py       # NEW: LLM integration
│   │   ├── conversation.py     # NEW: Conversation management
│   │   ├── symptom_analyzer.py # NEW: AI symptom extraction
│   │   ├── recommendation.py   # NEW: Health recommendations
│   │   ├── speech.py           # NEW: STT/TTS integration
│   │   └── prompts.py          # NEW: System prompts for AI
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   └── utils/
│       ├── __init__.py
│       ├── json_handler.py
│       └── logger.py
├── data/
│   ├── medication_schedule.json
│   ├── health_logs.json
│   ├── user_interactions.json
│   ├── conversations/          # NEW: Conversation transcripts
│   └── voice_recordings/       # NEW: Audio files (optional)
├── hardware/
│   └── pin_config.json
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_hardware.py
│   └── test_ai.py              # NEW: AI tests
├── requirements.txt
├── .env
└── README.md
```

#### Step 1.2: Install Dependencies

**For Python with FastAPI + AI:**

```bash
pip install fastapi uvicorn[standard] pydantic pydantic-settings httpx python-dotenv RPi.GPIO gpiozero python-multipart aiofiles openai anthropic spacy torch transformers
```

**Requirements.txt:**

```
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.0
pydantic-settings==2.1.0
httpx==0.26.0
python-dotenv==1.0.0
RPi.GPIO==0.7.1
gpiozero==2.0.1
python-multipart==0.0.6
aiofiles==23.2.1

# AI/ML Dependencies
openai==1.10.0              # OpenAI API client
anthropic==0.18.0           # Claude API client
spacy==3.7.0                # NLP for symptom extraction
transformers==4.36.0        # Hugging Face models
torch==2.1.0                # PyTorch for local models
openai-whisper==20231117    # Local Whisper for STT
TTS==0.21.0                 # Coqui TTS for local TTS
langchain==0.1.0            # LLM orchestration (optional)
```

### Phase 2: Hardware Interface Layer

#### Step 2.1: Define Hardware Pin Configuration

Create `hardware/pin_config.json`:

```json
{
  "servo_interface": {
    "type": "http",
    "base_url": "http://localhost:8080",
    "endpoints": {
      "open": "/servo/open",
      "close": "/servo/close",
      "status": "/servo/status"
    },
    "timeout": 5
  },
  "leds": {
    "compartment_1": 17,
    "compartment_2": 27,
    "compartment_3": 22,
    "compartment_4": 23,
    "status_led": 24
  }
}
```

#### Step 2.2: Create Servo Interface

File: `server/controllers/servo_interface.py`

**Requirements:**

- Async function to send open command to external servo controller (using httpx.AsyncClient)
- Async function to send close command to external servo controller
- Async function to check servo status from external controller
- Error handling for communication failures with proper async exception handling
- Retry logic for failed requests (using tenacity library or custom retry)
- Timeout handling with httpx timeout configuration
- Logging of all servo interface actions with timestamps

**Key Functions:**

```python
async def send_open_command(compartment_id: int) -> dict
async def send_close_command() -> dict
async def get_servo_status() -> dict
async def test_connection() -> bool
```

**Communication Protocol:**

- Async HTTP POST request to servo controller with JSON payload using httpx
- Request format: `{ "action": "open", "compartment": 1 }`
- Response format: `{ "status": "success/error", "position": "open/closed", "message": "..." }`
- Use httpx.AsyncClient as a singleton/dependency injection pattern
- Implement connection pooling for better performance

#### Step 2.3: Create LED Controller

File: `server/controllers/led_controller.py`

**Requirements:**

- Function to initialize all LED pins as outputs
- Function to turn on specific compartment LED
- Function to turn off all LEDs
- Async function to blink LED (for notifications) - use asyncio.sleep
- Function to show status pattern (e.g., blinking for error states)
- Thread-safe GPIO access (use asyncio locks if needed)

**Key Functions:**

```python
def initialize_leds(led_pins: dict) -> None
def highlight_compartment(compartment_number: int) -> None
def clear_all_leds() -> None
async def blink_led(pin: int, duration: float, frequency: float) -> None
def show_error_pattern() -> None
def show_success_pattern() -> None
```

### Phase 3: Data Models

#### Step 3.1: Define Pydantic Schemas

File: `server/models/schemas.py`

**Use Pydantic v2 models for automatic validation and serialization:**

```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime
from uuid import UUID, uuid4

class MedicationEntry(BaseModel):
    medication_id: UUID = Field(default_factory=uuid4)
    name: str
    compartment: int = Field(ge=1, le=4)
    scheduled_time: datetime
    taken: bool = False
    taken_timestamp: Optional[datetime] = None

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "name": "Blood Pressure Medication",
            "compartment": 1,
            "scheduled_time": "2024-01-15T08:00:00Z"
        }
    })

class HealthSymptoms(BaseModel):
    pain_level: int = Field(ge=0, le=10)
    nausea: bool = False
    dizziness: bool = False
    fatigue: bool = False
    custom_notes: Optional[str] = None

class VitalSigns(BaseModel):
    mood: str = Field(pattern="^(good|okay|bad)$")
    sleep_quality: str = Field(pattern="^(good|fair|poor)$")

class AIInteraction(BaseModel):
    questions_asked: List[str]
    responses_given: List[str]
    conversation_summary: Optional[str] = None

class HealthDataEntry(BaseModel):
    entry_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    medication_id: Optional[UUID] = None
    symptoms: HealthSymptoms
    vital_signs: VitalSigns
    ai_interaction: Optional[AIInteraction] = None

class DispensingEvent(BaseModel):
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    compartment: int = Field(ge=1, le=4)
    medication_id: UUID
    status: str = Field(pattern="^(success|failed|skipped)$")
    box_opened: bool
    led_activated: bool
    servo_response: Optional[str] = None
    error_message: Optional[str] = None

class DispenseRequest(BaseModel):
    compartment: int = Field(ge=1, le=4)
    medication_id: UUID

class MarkTakenRequest(BaseModel):
    medication_id: UUID
    timestamp: Optional[datetime] = None

# NEW: AI Conversation Models
class ConversationMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ConversationSession(BaseModel):
    session_id: UUID = Field(default_factory=uuid4)
    user_id: str
    medication_id: Optional[UUID] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    messages: List[ConversationMessage] = []
    extracted_symptoms: Optional[dict] = None
    sentiment_score: Optional[float] = None  # -1 to 1

class VoiceInteractionRequest(BaseModel):
    audio_data: Optional[str] = None  # Base64 encoded audio
    text_input: Optional[str] = None  # Alternative text input
    medication_id: Optional[UUID] = None
    language: str = "en"

class VoiceInteractionResponse(BaseModel):
    session_id: UUID
    ai_response_text: str
    ai_response_audio: Optional[str] = None  # Base64 encoded audio
    extracted_symptoms: Optional[dict] = None
    recommendations: Optional[List[str]] = None
    follow_up_questions: Optional[List[str]] = None

class SymptomExtractionResult(BaseModel):
    symptoms: dict  # Structured symptom data
    confidence_scores: dict
    extracted_entities: List[str]
    sentiment: str = Field(pattern="^(positive|neutral|negative)$")
    urgency_level: str = Field(pattern="^(low|medium|high|urgent)$")

class HealthRecommendation(BaseModel):
    recommendation_id: UUID = Field(default_factory=uuid4)
    recommendation_text: str
    category: str  # "lifestyle", "medication", "seek_help", etc.
    priority: str = Field(pattern="^(low|medium|high)$")
    based_on: List[str]  # List of symptoms/data points
    generated_at: datetime = Field(default_factory=datetime.utcnow)
```

#### Step 3.2: Configuration with Pydantic Settings

File: `server/config.py`

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Server settings
    app_name: str = "Krab Med Bot API"
    host: str = "0.0.0.0"
    port: int = 5000

    # Hardware settings
    hardware_mode: str = "production"  # or "mock"
    servo_controller_url: str = "http://localhost:8080"
    servo_timeout: int = 5

    # Data settings
    data_dir: str = "./data"
    log_level: str = "INFO"

    # CORS settings
    cors_origins: List[str] = ["http://localhost:3000"]

    # AI/LLM Settings
    ai_provider: str = "openai"  # "openai", "anthropic", "ollama"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # LLM Model Settings
    llm_model: str = "gpt-4-turbo-preview"  # or "claude-3-opus-20240229"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 500

    # Speech Settings
    stt_provider: str = "openai"  # "openai", "whisper_local"
    tts_provider: str = "openai"  # "openai", "elevenlabs", "local"
    tts_voice: str = "alloy"  # OpenAI TTS voice

    # Conversation Settings
    conversation_context_window: int = 10  # Last N messages to include
    enable_voice_interaction: bool = True
    auto_symptom_extraction: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

settings = Settings()
```

### Phase 4: AI Integration Layer

#### Step 4.1: LLM Client

File: `server/ai/llm_client.py`

**Requirements:**

- Async client for OpenAI/Anthropic/Ollama
- Support multiple LLM providers with unified interface
- Token counting and rate limiting
- Error handling and retry logic
- Streaming support for real-time responses
- Cost tracking for API usage

**Key Functions:**

```python
import openai
import anthropic
import httpx
from typing import AsyncIterator, List, Dict

class LLMClient:
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        if provider == "openai":
            self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        elif provider == "anthropic":
            self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        elif provider == "ollama":
            self.base_url = settings.ollama_base_url

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """Generate AI response from conversation history"""
        pass

    async def stream_response(
        self,
        messages: List[Dict[str, str]]
    ) -> AsyncIterator[str]:
        """Stream AI response token by token"""
        pass

    async def count_tokens(self, text: str) -> int:
        """Count tokens in text for cost estimation"""
        pass

llm_client = LLMClient(provider=settings.ai_provider)
```

#### Step 4.2: Conversation Manager

File: `server/ai/conversation.py`

**Requirements:**

- Manage conversation sessions with context
- Store and retrieve conversation history
- Build conversation context with system prompts
- Handle multi-turn conversations
- Track conversation state (greeting, questioning, closing)
- Implement conversation memory management

**Key Functions:**

```python
from server.models.schemas import ConversationSession, ConversationMessage
from server.ai.prompts import get_system_prompt

class ConversationManager:
    def __init__(self):
        self.active_sessions: Dict[UUID, ConversationSession] = {}

    async def start_session(
        self,
        user_id: str,
        medication_id: Optional[UUID] = None
    ) -> ConversationSession:
        """Start a new conversation session"""
        session = ConversationSession(
            user_id=user_id,
            medication_id=medication_id
        )

        # Add system prompt
        system_message = ConversationMessage(
            role="system",
            content=get_system_prompt(medication_id)
        )
        session.messages.append(system_message)

        self.active_sessions[session.session_id] = session
        return session

    async def add_message(
        self,
        session_id: UUID,
        role: str,
        content: str
    ) -> ConversationMessage:
        """Add a message to the conversation"""
        pass

    async def get_context(
        self,
        session_id: UUID,
        max_messages: int = 10
    ) -> List[Dict[str, str]]:
        """Get recent conversation context for LLM"""
        pass

    async def generate_response(
        self,
        session_id: UUID,
        user_message: str
    ) -> str:
        """Generate AI response to user message"""
        # Add user message
        await self.add_message(session_id, "user", user_message)

        # Get context
        context = await self.get_context(session_id)

        # Generate response
        ai_response = await llm_client.generate_response(context)

        # Add AI response
        await self.add_message(session_id, "assistant", ai_response)

        return ai_response

    async def end_session(self, session_id: UUID) -> ConversationSession:
        """End conversation and save to storage"""
        pass

conversation_manager = ConversationManager()
```

#### Step 4.3: System Prompts

File: `server/ai/prompts.py`

**Requirements:**

- Define system prompts for different conversation contexts
- Include elderly-friendly language guidelines
- Specify symptom extraction instructions
- Define safety guidelines and constraints

**System Prompts:**

```python
def get_system_prompt(medication_id: Optional[UUID] = None) -> str:
    """
    Get the system prompt for medication check-in conversations
    """
    return """You are a caring and patient AI assistant helping elderly users with their medication routine.

Your role:
- Greet the user warmly and ask how they're feeling
- Ask simple, clear questions about their health
- Listen carefully for any symptoms or side effects
- Provide gentle encouragement and reassurance
- Suggest simple self-care tips when appropriate
- Always be patient and speak clearly

Guidelines:
- Use simple language, avoid medical jargon
- Ask one question at a time
- Be empathetic and supportive
- If the user reports severe symptoms, advise them to contact their doctor
- Keep responses brief and conversational
- Remember you're talking to an elderly person who may need things repeated

Current context:
- The user is taking their medication now
- Ask about: current symptoms, pain levels, sleep quality, mood, appetite
- Listen for: dizziness, nausea, unusual tiredness, confusion, pain

Example questions:
- "Good morning! How are you feeling today?"
- "Did you sleep well last night?"
- "Are you experiencing any pain or discomfort?"
- "How is your appetite today?"

Never:
- Diagnose medical conditions
- Suggest changing medication dosage
- Provide emergency medical advice
- Be judgmental or dismissive

Always:
- Be warm and friendly
- Validate their feelings
- Encourage them to contact their doctor if needed
- End with a positive note
"""

def get_symptom_extraction_prompt(conversation: str) -> str:
    """
    Prompt for extracting structured symptom data from conversation
    """
    return f"""Analyze this conversation between an elderly patient and an AI assistant.
Extract all mentioned symptoms, health indicators, and vital signs.

Conversation:
{conversation}

Extract and structure the following information as JSON:
{{
    "symptoms": {{
        "pain_level": 0-10 or null,
        "nausea": true/false,
        "dizziness": true/false,
        "fatigue": true/false,
        "headache": true/false,
        "shortness_of_breath": true/false,
        "other_symptoms": []
    }},
    "vital_signs": {{
        "mood": "good/okay/bad" or null,
        "sleep_quality": "good/fair/poor" or null,
        "appetite": "good/fair/poor" or null
    }},
    "concerns": [],
    "urgency_level": "low/medium/high/urgent",
    "notes": "brief summary"
}}

Be conservative - only include information explicitly mentioned by the patient.
If uncertain, mark as null or omit.
"""

def get_recommendation_prompt(symptoms: dict, user_history: dict) -> str:
    """
    Prompt for generating personalized health recommendations
    """
    return f"""Based on the following patient information, provide gentle, actionable health recommendations.

Current symptoms: {symptoms}
Recent history: {user_history}

Provide 2-4 simple recommendations that are:
- Easy for an elderly person to implement
- Safe and non-medical (lifestyle tips only)
- Specific and actionable
- Encouraging and positive

Categories:
- Hydration
- Rest
- Gentle movement
- Nutrition
- When to seek help

Format each recommendation as:
- Clear action item
- Why it might help
- How to do it simply

If symptoms suggest urgency, include a recommendation to contact their doctor.
"""
```

#### Step 4.4: Symptom Analyzer

File: `server/ai/symptom_analyzer.py`

**Requirements:**

- Extract symptoms from natural conversation
- Use NLP to identify health-related entities
- Classify symptom severity and urgency
- Track symptom trends over time
- Generate confidence scores

**Key Functions:**

```python
import spacy
from typing import Dict, List
import json

class SymptomAnalyzer:
    def __init__(self):
        # Load spaCy model for entity extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # Download if not available
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

    async def extract_symptoms(
        self,
        conversation_text: str
    ) -> SymptomExtractionResult:
        """
        Extract symptoms from conversation using NLP and LLM
        """
        # Use NLP for entity extraction
        doc = self.nlp(conversation_text)
        entities = [ent.text for ent in doc.ents]

        # Use LLM for structured extraction
        extraction_prompt = get_symptom_extraction_prompt(conversation_text)
        llm_response = await llm_client.generate_response([
            {"role": "system", "content": "You are a medical data extraction assistant."},
            {"role": "user", "content": extraction_prompt}
        ])

        # Parse LLM response
        symptoms_data = json.loads(llm_response)

        # Calculate urgency
        urgency = self._calculate_urgency(symptoms_data)

        return SymptomExtractionResult(
            symptoms=symptoms_data.get("symptoms", {}),
            confidence_scores={},  # TODO: Implement confidence scoring
            extracted_entities=entities,
            sentiment=self._analyze_sentiment(conversation_text),
            urgency_level=urgency
        )

    def _calculate_urgency(self, symptoms: dict) -> str:
        """Calculate urgency level from symptoms"""
        # Check for urgent symptoms
        urgent_symptoms = [
            "chest_pain", "difficulty_breathing", "severe_pain",
            "confusion", "sudden_weakness"
        ]

        pain_level = symptoms.get("symptoms", {}).get("pain_level", 0)

        if pain_level >= 8:
            return "urgent"
        elif pain_level >= 6:
            return "high"
        elif pain_level >= 3:
            return "medium"
        else:
            return "low"

    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of conversation"""
        # Simple keyword-based sentiment (could use transformers for better results)
        positive_words = ["good", "better", "fine", "well", "great"]
        negative_words = ["bad", "worse", "pain", "dizzy", "nausea", "tired"]

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"

symptom_analyzer = SymptomAnalyzer()
```

#### Step 4.5: Health Recommendations

File: `server/ai/recommendation.py`

**Requirements:**

- Generate personalized recommendations based on symptoms
- Consider user history and trends
- Provide actionable, elderly-friendly advice
- Prioritize recommendations by urgency
- Track recommendation effectiveness

**Key Functions:**

```python
from server.models.schemas import HealthRecommendation
from server.ai.prompts import get_recommendation_prompt

class RecommendationEngine:
    async def generate_recommendations(
        self,
        symptoms: dict,
        user_history: dict
    ) -> List[HealthRecommendation]:
        """
        Generate personalized health recommendations
        """
        # Build context
        prompt = get_recommendation_prompt(symptoms, user_history)

        # Get LLM recommendations
        llm_response = await llm_client.generate_response([
            {"role": "system", "content": "You are a helpful health advisor for elderly patients."},
            {"role": "user", "content": prompt}
        ])

        # Parse and structure recommendations
        recommendations = self._parse_recommendations(llm_response, symptoms)

        return recommendations

    def _parse_recommendations(
        self,
        llm_text: str,
        symptoms: dict
    ) -> List[HealthRecommendation]:
        """Parse LLM text into structured recommendations"""
        # Split by bullet points or numbers
        lines = [line.strip() for line in llm_text.split('\n') if line.strip()]

        recommendations = []
        for line in lines:
            if line.startswith('-') or line.startswith('•') or line[0].isdigit():
                # Clean the line
                clean_line = line.lstrip('-•0123456789. ')

                # Categorize
                category = self._categorize_recommendation(clean_line)
                priority = self._assess_priority(clean_line, symptoms)

                rec = HealthRecommendation(
                    recommendation_text=clean_line,
                    category=category,
                    priority=priority,
                    based_on=list(symptoms.keys())
                )
                recommendations.append(rec)

        return recommendations

    def _categorize_recommendation(self, text: str) -> str:
        """Categorize recommendation"""
        text_lower = text.lower()
        if any(word in text_lower for word in ["water", "drink", "hydrate"]):
            return "hydration"
        elif any(word in text_lower for word in ["rest", "sleep", "relax"]):
            return "rest"
        elif any(word in text_lower for word in ["walk", "exercise", "move"]):
            return "activity"
        elif any(word in text_lower for word in ["doctor", "physician", "medical"]):
            return "seek_help"
        else:
            return "general"

    def _assess_priority(self, text: str, symptoms: dict) -> str:
        """Assess recommendation priority"""
        text_lower = text.lower()
        if any(word in text_lower for word in ["immediately", "urgent", "doctor", "emergency"]):
            return "high"
        elif any(word in text_lower for word in ["today", "soon", "important"]):
            return "medium"
        else:
            return "low"

recommendation_engine = RecommendationEngine()
```

#### Step 4.6: Speech-to-Text and Text-to-Speech

File: `server/ai/speech.py`

**Requirements:**

- Convert audio to text (STT) for voice interaction
- Convert AI responses to speech (TTS)
- Support multiple languages
- Handle audio file formats
- Optimize for elderly speech patterns

**Key Functions:**

```python
import base64
import io
from openai import AsyncOpenAI
import aiofiles

class SpeechService:
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def speech_to_text(
        self,
        audio_data: str,  # Base64 encoded
        language: str = "en"
    ) -> str:
        """
        Convert audio to text using Whisper API
        """
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data)

        # Create file-like object
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav"

        # Use OpenAI Whisper
        transcription = await self.openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language
        )

        return transcription.text

    async def text_to_speech(
        self,
        text: str,
        voice: str = "alloy"  # alloy, echo, fable, onyx, nova, shimmer
    ) -> str:
        """
        Convert text to speech and return base64 encoded audio
        """
        # Use OpenAI TTS
        response = await self.openai_client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )

        # Get audio bytes
        audio_bytes = response.content

        # Encode to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        return audio_base64

    async def save_audio(
        self,
        audio_data: str,
        filename: str
    ) -> str:
        """Save audio file to disk"""
        filepath = f"data/voice_recordings/{filename}"
        audio_bytes = base64.b64decode(audio_data)

        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(audio_bytes)

        return filepath

speech_service = SpeechService()
```

### Phase 5: API Endpoints with AI

#### Step 5.1: AI Conversation Endpoints

File: `server/routes/ai_conversation.py`

```python
from fastapi import APIRouter, HTTPException
from server.models.schemas import (
    ConversationSession,
    ConversationMessage,
    VoiceInteractionRequest,
    VoiceInteractionResponse
)
from server.ai.conversation import conversation_manager
from server.ai.symptom_analyzer import symptom_analyzer
from server.ai.recommendation import recommendation_engine
from uuid import UUID

router = APIRouter(prefix="/api/ai", tags=["ai"])

@router.post("/conversation/start", response_model=ConversationSession)
async def start_conversation(
    user_id: str,
    medication_id: Optional[UUID] = None
):
    """
    Start a new AI conversation session for medication check-in
    """
    session = await conversation_manager.start_session(user_id, medication_id)

    # Generate opening greeting
    greeting = await conversation_manager.generate_response(
        session.session_id,
        "User has opened the medication box."  # Internal trigger
    )

    return session

@router.post("/conversation/{session_id}/message", response_model=ConversationMessage)
async def send_message(
    session_id: UUID,
    message: str
):
    """
    Send a message in an ongoing conversation
    """
    ai_response = await conversation_manager.generate_response(
        session_id,
        message
    )

    return ConversationMessage(
        role="assistant",
        content=ai_response
    )

@router.post("/conversation/{session_id}/end")
async def end_conversation(session_id: UUID):
    """
    End conversation and extract symptoms/generate recommendations
    """
    session = await conversation_manager.end_session(session_id)

    # Extract conversation text
    conversation_text = "\n".join([
        f"{msg.role}: {msg.content}"
        for msg in session.messages
        if msg.role != "system"
    ])

    # Extract symptoms
    symptoms = await symptom_analyzer.extract_symptoms(conversation_text)

    # Generate recommendations
    recommendations = await recommendation_engine.generate_recommendations(
        symptoms.symptoms,
        {}  # TODO: Get user history
    )

    # Save to health logs
    health_entry = HealthDataEntry(
        medication_id=session.medication_id,
        symptoms=symptoms.symptoms,
        vital_signs={},  # Extracted from symptoms
        ai_interaction=AIInteraction(
            questions_asked=[m.content for m in session.messages if m.role == "assistant"],
            responses_given=[m.content for m in session.messages if m.role == "user"],
            conversation_summary=f"Extracted {len(symptoms.symptoms)} symptoms"
        )
    )

    # TODO: Save health_entry to data/health_logs.json

    return {
        "session_id": session_id,
        "symptoms": symptoms,
        "recommendations": recommendations,
        "urgency_level": symptoms.urgency_level
    }

@router.get("/conversation/{session_id}")
async def get_conversation(session_id: UUID):
    """Get conversation details"""
    if session_id not in conversation_manager.active_sessions:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return conversation_manager.active_sessions[session_id]
```

#### Step 5.2: Voice Interaction Endpoints

File: `server/routes/voice.py`

```python
from fastapi import APIRouter, UploadFile, File
from server.models.schemas import VoiceInteractionRequest, VoiceInteractionResponse
from server.ai.speech import speech_service
from server.ai.conversation import conversation_manager
from server.ai.symptom_analyzer import symptom_analyzer
import base64

router = APIRouter(prefix="/api/voice", tags=["voice"])

@router.post("/interact", response_model=VoiceInteractionResponse)
async def voice_interaction(request: VoiceInteractionRequest):
    """
    Handle voice interaction with AI
    - Accepts audio or text input
    - Returns AI response as audio and text
    - Extracts symptoms automatically
    """
    # Convert audio to text if provided
    if request.audio_data:
        user_text = await speech_service.speech_to_text(
            request.audio_data,
            language=request.language
        )
    else:
        user_text = request.text_input

    # Start or continue conversation
    # For simplicity, create new session each time (could track session)
    session = await conversation_manager.start_session("voice_user", request.medication_id)

    # Generate AI response
    ai_text = await conversation_manager.generate_response(
        session.session_id,
        user_text
    )

    # Convert response to speech
    ai_audio = None
    if settings.enable_voice_interaction:
        ai_audio = await speech_service.text_to_speech(ai_text, voice=settings.tts_voice)

    # Extract symptoms if auto-extraction enabled
    extracted_symptoms = None
    recommendations = None
    if settings.auto_symptom_extraction:
        conversation_text = f"User: {user_text}\nAI: {ai_text}"
        symptom_result = await symptom_analyzer.extract_symptoms(conversation_text)
        extracted_symptoms = symptom_result.symptoms

        if extracted_symptoms:
            recommendations_list = await recommendation_engine.generate_recommendations(
                extracted_symptoms,
                {}
            )
            recommendations = [r.recommendation_text for r in recommendations_list]

    # Generate follow-up questions
    follow_ups = [
        "How is your pain level on a scale of 1 to 10?",
        "Did you sleep well last night?",
        "Are you experiencing any dizziness?"
    ]

    return VoiceInteractionResponse(
        session_id=session.session_id,
        ai_response_text=ai_text,
        ai_response_audio=ai_audio,
        extracted_symptoms=extracted_symptoms,
        recommendations=recommendations,
        follow_up_questions=follow_ups[:2]  # Limit to 2
    )

@router.post("/upload-audio")
async def upload_audio_file(file: UploadFile = File(...)):
    """
    Upload audio file for transcription
    Alternative to base64 encoding
    """
    audio_bytes = await file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

    transcription = await speech_service.speech_to_text(audio_base64)

    return {"transcription": transcription}
```

#### Step 5.3: Enhanced Hardware Endpoints with AI

File: `server/routes/hardware.py`

**Update dispense endpoint to trigger AI conversation:**

```python
// ...existing code...

@router.post("/dispense", response_model=DispensingEvent)
async def dispense_medication(request: DispenseRequest):
    """
    Dispense medication from specified compartment
    - Activates LED
    - Sends command to servo controller
    - Logs event
    - TRIGGERS AI CONVERSATION (NEW)
    """
    # Existing dispensing logic
    led_controller.highlight_compartment(request.compartment)
    servo_response = await servo_interface.send_open_command(request.compartment)

    # NEW: Start AI conversation automatically
    session = await conversation_manager.start_session(
        user_id="current_user",  # TODO: Get from auth
        medication_id=request.medication_id
    )

    # Generate greeting
    greeting = await conversation_manager.generate_response(
        session.session_id,
        "User is taking medication now."
    )

    # Return dispense event with conversation session
    return DispensingEvent(
        compartment=request.compartment,
        medication_id=request.medication_id,
        status="success",
        box_opened=True,
        led_activated=True,
        servo_response=str(servo_response),
        conversation_session_id=session.session_id,  # NEW field
        ai_greeting=greeting  # NEW field
    )

// ...existing code...
```

### Phase 10: Additional Features

#### Step 10.1: Scheduled Tasks

Use cron or APScheduler for:

- Automatic medication reminders **with AI-generated personalized messages**
- Daily health summary generation **with AI trend analysis**
- Data backup
- Log file rotation
- Servo controller health checks
- **AI conversation summary and insights**
- **Proactive health alerts based on symptom trends**

#### Step 10.2: AI-Powered Safety Features

- **Urgency detection**: AI analyzes symptoms for urgent conditions
- **Trend analysis**: Track symptom patterns over time
- **Caregiver alerts**: Automatically notify caregivers if concerning patterns detected
- **Medication interaction warnings**: AI checks for potential issues (future)
- **Fall detection integration**: Connect with sensors for comprehensive care
- **Emergency escalation**: Auto-suggest calling doctor for severe symptoms

#### Step 10.3: Advanced AI Features

- **Multimodal interaction**: Support text, voice, and video
- **Emotion detection**: Analyze tone and sentiment for mental health insights
- **Personalized conversation style**: Adapt to user's preferences over time
- **Multilingual support**: Conversations in user's preferred language
- **Memory across sessions**: Remember previous conversations and trends
- **Predictive health modeling**: Predict potential issues before they occur
- **Integration with wearables**: Incorporate data from smartwatches, etc.

## Implementation Order

1. **Day 1**: Project setup, install FastAPI + AI libraries, create basic app structure, config
2. **Day 2**: LED controller, async servo interface, basic LLM client
3. **Day 3**: Pydantic models (including AI models), conversation manager, system prompts
4. **Day 4**: Hardware API endpoints, AI conversation endpoints
5. **Day 5**: Speech services (STT/TTS), voice interaction endpoints
6. **Day 6**: Symptom analyzer, recommendation engine, health data endpoints
7. **Day 7**: Testing, dashboard with AI insights, documentation, deployment

## Testing Without Hardware or AI APIs

**Mock mode with environment variable:**

```python
# server/ai/llm_client.py
async def generate_response(self, messages: List[Dict]) -> str:
    if settings.ai_provider == "mock":
        # Return predefined responses for testing
        return "Thank you for sharing. How are you feeling today?"
    else:
        # Actual API call
        pass

# server/ai/speech.py
async def speech_to_text(self, audio_data: str) -> str:
    if settings.stt_provider == "mock":
        return "I'm feeling okay, just a little tired."
    else:
        # Actual STT
        pass

async def text_to_speech(self, text: str) -> str:
    if settings.tts_provider == "mock":
        # Return empty audio or sample
        return base64.b64encode(b"mock_audio").decode()
    else:
        # Actual TTS
        pass
```

## Quick Start Commands

```bash
# Clone and setup
cd /c:/Users/bruce/Projects/krab_med_bot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Create necessary directories
mkdir -p data logs hardware tests data/conversations data/voice_recordings

# Create .env file with AI credentials
cat > .env << EOF
HARDWARE_MODE=mock
SERVO_CONTROLLER_URL=http://localhost:8080
PORT=5000

# AI Settings
AI_PROVIDER=openai
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-4-turbo-preview
LLM_TEMPERATURE=0.7

# Speech Settings
STT_PROVIDER=openai
TTS_PROVIDER=openai
TTS_VOICE=alloy

# Features
ENABLE_VOICE_INTERACTION=true
AUTO_SYMPTOM_EXTRACTION=true
EOF

# Run in development mode with auto-reload
uvicorn server.main:app --reload --host 0.0.0.0 --port 5000

# Access API documentation
# Open browser: http://localhost:5000/docs

# Test AI endpoints
curl -X POST http://localhost:5000/api/ai/conversation/start \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user", "medication_id": null}'

# Test voice interaction (with text fallback)
curl -X POST http://localhost:5000/api/voice/interact \
  -H "Content-Type: application/json" \
  -d '{"text_input": "I have a headache", "medication_id": null}'

# Run tests
pytest tests/ -v --asyncio-mode=auto
```

## AI Integration Advantages for This Project

1. **Natural Conversation**: Elderly users can talk naturally instead of filling forms
2. **Empathetic Interaction**: AI provides emotional support and encouragement
3. **Automatic Data Collection**: Symptoms extracted without explicit questions
4. **Personalized Care**: Recommendations tailored to individual patterns
5. **Proactive Monitoring**: Early detection of concerning trends
6. **Reduced Caregiver Burden**: Automated check-ins with intelligent escalation
7. **Accessibility**: Voice interaction removes barriers for those with limited mobility
8. **Continuous Learning**: System improves with more conversations
9. **24/7 Availability**: AI always available for questions and support
10. **Comprehensive Insights**: Track health holistically over time
