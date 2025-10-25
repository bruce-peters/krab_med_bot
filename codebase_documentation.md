# Krab Med Bot - Complete Codebase Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Core Components](#core-components)
5. [API Endpoints](#api-endpoints)
6. [Data Models](#data-models)
7. [Configuration](#configuration)
8. [Setup & Usage](#setup--usage)
9. [Testing](#testing)
10. [Deployment](#deployment)

---

## Project Overview

**Krab Med Bot** is an AI-powered medication dispensing system designed for elderly users. It combines hardware control (LEDs, servo motors) with intelligent conversational AI to:

- Dispense medications from a 4-compartment box
- Conduct health check-ins via voice/text conversations
- Automatically extract symptoms from natural conversations
- Generate personalized health recommendations
- Track medication adherence and health trends
- Provide caregiver insights and alerts

### Key Features

- **Hardware Integration**: Controls LEDs and servo motors via HTTP interface
- **AI Conversations**: Natural language interactions using OpenAI/Claude/Ollama
- **Voice Support**: Speech-to-text and text-to-speech capabilities
- **Symptom Analysis**: Automatic health data extraction using NLP
- **Smart Recommendations**: AI-generated health advice based on symptoms
- **Real-time Updates**: WebSocket support for live notifications
- **Mock Mode**: Full testing without physical hardware or API keys

---

## Architecture

```
┌─────────────┐
│   Frontend  │ (React/HTML)
└──────┬──────┘
       │ HTTP/WebSocket
       ▼
┌─────────────────────────────────────┐
│      FastAPI Server (Python)        │
│  ┌─────────────────────────────┐   │
│  │  Routes (API Endpoints)     │   │
│  ├─────────────────────────────┤   │
│  │  AI Layer                   │   │
│  │  - LLM Client               │   │
│  │  - Conversation Manager     │   │
│  │  - Symptom Analyzer         │   │
│  │  - Speech Services          │   │
│  ├─────────────────────────────┤   │
│  │  Hardware Interface         │   │
│  │  - LED Control              │   │
│  │  - Servo Communication      │   │
│  └─────────────────────────────┘   │
└──────┬────────────────────┬─────────┘
       │                    │
       ▼                    ▼
┌──────────────┐    ┌──────────────┐
│ External     │    │ OpenAI/      │
│ Hardware     │    │ Claude API   │
│ Controller   │    │              │
└──────────────┘    └──────────────┘
       │
       ▼
┌──────────────┐
│ Physical     │
│ LEDs/Servos  │
└──────────────┘
```

---

## Directory Structure

```
krab_med_bot/
├── server/                          # Backend application
│   ├── main.py                      # FastAPI app entry point
│   ├── config.py                    # Configuration management
│   │
│   ├── routes/                      # API endpoint definitions
│   │   ├── __init__.py
│   │   ├── hardware.py              # Hardware control endpoints
│   │   ├── medication.py            # Medication management
│   │   ├── health_data.py           # Health logging endpoints
│   │   ├── ai_conversation.py       # AI chat endpoints
│   │   └── voice.py                 # Voice interaction endpoints
│   │
│   ├── controllers/                 # Hardware control layer
│   │   ├── __init__.py
│   │   └── hardware_interface.py   # Unified hardware interface
│   │
│   ├── ai/                          # AI/ML components
│   │   ├── __init__.py
│   │   ├── llm_client.py           # LLM API client
│   │   ├── conversation.py         # Conversation manager
│   │   ├── symptom_analyzer.py     # Symptom extraction
│   │   ├── recommendation.py       # Health recommendations
│   │   ├── speech.py               # STT/TTS services
│   │   └── prompts.py              # AI system prompts
│   │
│   ├── models/                      # Data models
│   │   ├── __init__.py
│   │   └── schemas.py              # Pydantic models
│   │
│   └── utils/                       # Utility functions
│       ├── __init__.py
│       ├── json_handler.py         # JSON file operations
│       └── logger.py               # Logging configuration
│
├── data/                            # Data storage (JSON files)
│   ├── medication_schedule.json    # Medication schedules
│   ├── health_logs.json            # Health data entries
│   ├── user_interactions.json      # User activity logs
│   ├── dispensing_events.json      # Dispensing history
│   ├── conversations/              # AI conversation transcripts
│   └── voice_recordings/           # Audio files (optional)
│
├── hardware/                        # Hardware configuration
│   └── hardware_config.json        # Pin mappings, endpoints
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── test_api.py                 # API endpoint tests
│   ├── test_hardware.py            # Hardware interface tests
│   └── test_ai.py                  # AI component tests
│
├── requirements.txt                 # Python dependencies
├── .env                            # Environment variables
├── Implementation.md               # Implementation guide
└── README.md                       # Project README
```

---

## Core Components

### 1. Main Application (`server/main.py`)

**Purpose**: FastAPI application entry point, sets up routes, middleware, and lifecycle events.

**Key Features**:

- CORS middleware configuration
- Route registration (hardware, medication, health, AI, voice)
- Hardware interface initialization
- Startup/shutdown event handlers
- Health check endpoint

**Important Code**:

```python
@app.on_event("startup")
async def startup_event():
    """Initialize hardware interface on startup"""
    await app.state.hardware.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up hardware interface on shutdown"""
    await app.state.hardware.close()
```

**Endpoints Provided**:

- `GET /` - Root endpoint with API info
- `GET /health` - Health check

---

### 2. Configuration (`server/config.py`)

**Purpose**: Centralized configuration using Pydantic Settings with environment variable support.

**Configuration Categories**:

1. **Server Settings**

   - `app_name`: Application name
   - `host`: Server host (default: 0.0.0.0)
   - `port`: Server port (default: 5000)

2. **Hardware Settings**

   - `hardware_mode`: "production" or "mock"
   - `servo_controller_url`: External hardware controller URL
   - `servo_timeout`: Timeout for hardware requests

3. **AI/LLM Settings**

   - `ai_provider`: "openai", "anthropic", "gemini", or "ollama"
   - `openai_api_key`: OpenAI API key
   - `anthropic_api_key`: Anthropic API key
   - `gemini_api_key`: Google Gemini API key
   - `llm_model`: Model name (e.g., "gpt-4-turbo-preview", "gemini-1.5-flash")
   - `llm_temperature`: Response randomness (0-1)
   - `llm_max_tokens`: Maximum response length

4. **Speech Settings**

   - `stt_provider`: Speech-to-text provider ("openai", "gemini", "whisper_local", "mock")
   - `tts_provider`: Text-to-speech provider ("openai", "gemini", "elevenlabs", "local", "mock")
   - `tts_voice`: Voice selection for TTS

**Speech Providers**:

- **Gemini**: Uses Gemini 2.0 Flash multimodal for audio transcription (STT only, TTS falls back to OpenAI)
- **OpenAI**: Whisper for STT, TTS-1 for TTS (high quality, paid)
- **Mock**: Testing mode with placeholder responses

5. **Feature Flags**
   - `enable_voice_interaction`: Enable/disable voice
   - `auto_symptom_extraction`: Auto-extract symptoms

**Usage**:

```python
from server.config import settings

# Access configuration
api_key = settings.openai_api_key
model = settings.llm_model
```

---

## API Endpoints

### Hardware Control (`/api/hardware`)

#### `POST /api/hardware/dispense`

**Purpose**: Dispense medication from specified compartment and start AI conversation.

**Request**:

```json
{
  "compartment": 1,
  "medication_id": "uuid-here"
}
```

**Response**:

```json
{
  "event_id": "uuid",
  "timestamp": "2024-01-15T10:30:00Z",
  "compartment": 1,
  "medication_id": "uuid",
  "status": "success",
  "box_opened": true,
  "led_activated": true,
  "conversation_session_id": "session-uuid",
  "ai_greeting": "Good morning! How are you feeling today?"
}
```

**Process Flow**:

1. Turn on LED for compartment
2. Send open command to servo
3. Start AI conversation session
4. Generate greeting message
5. Show success pattern on LED
6. Log dispensing event
7. Return event + conversation details

---

## Setup & Usage

### Prerequisites

- Python 3.9+
- pip
- Virtual environment (recommended)
- OpenAI API key (or other LLM provider)
- External hardware controller (or use mock mode)

### Installation

```bash
# Clone repository
cd /c:/Users/bruce/Projects/krab_med_bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model for NLP
python -m spacy download en_core_web_sm

# Create necessary directories
mkdir -p data/conversations data/voice_recordings hardware logs
```

### Configuration

Create `.env` file:

```bash
# Server
PORT=5000
HOST=0.0.0.0

# Hardware
HARDWARE_MODE=mock
SERVO_CONTROLLER_URL=http://localhost:8080

# AI
AI_PROVIDER=gemini
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
GEMINI_API_KEY=your-gemini-api-key-here
LLM_MODEL=gemini-1.5-flash
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=500

# Speech
STT_PROVIDER=gemini
TTS_PROVIDER=openai
TTS_VOICE=alloy

# Features
ENABLE_VOICE_INTERACTION=true
AUTO_SYMPTOM_EXTRACTION=true
CONVERSATION_CONTEXT_WINDOW=10
```

### Running the Server

```bash
# Development mode (with auto-reload)
uvicorn server.main:app --reload --host 0.0.0.0 --port 5000

# Production mode
uvicorn server.main:app --host 0.0.0.0 --port 5000 --workers 4
```

### API Documentation

Access interactive API docs:

- Swagger UI: `http://localhost:5000/docs`
- ReDoc: `http://localhost:5000/redoc`

---

**Last Updated**: January 2024
**Version**: 1.0.0
