"""
Pydantic Data Models for Krab Med Bot
All request/response schemas with validation
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4


# ========== MEDICATION MODELS ==========

class MedicationEntry(BaseModel):
    """Medication entry in the schedule"""
    medication_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=200)
    compartment: int = Field(..., ge=1, le=4, description="Compartment number (1-4)")
    scheduled_time: datetime
    taken: bool = False
    taken_timestamp: Optional[datetime] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "medication_id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Blood Pressure Medication",
                "compartment": 1,
                "scheduled_time": "2024-01-15T08:00:00Z",
                "taken": False,
                "taken_timestamp": None
            }
        }
    )


class MedicationSchedule(BaseModel):
    """Complete medication schedule for a user"""
    user_id: str
    medications: List[MedicationEntry] = []
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user_001",
                "medications": [
                    {
                        "name": "Morning Vitamin",
                        "compartment": 1,
                        "scheduled_time": "2024-01-15T08:00:00Z"
                    }
                ]
            }
        }
    )


# ========== HEALTH DATA MODELS ==========

class HealthSymptoms(BaseModel):
    """Symptoms reported by user"""
    pain_level: Optional[int] = Field(None, ge=0, le=10, description="Pain level 0-10")
    nausea: bool = False
    dizziness: bool = False
    fatigue: bool = False
    headache: bool = False
    shortness_of_breath: bool = False
    custom_notes: Optional[str] = Field(None, max_length=1000)
    other_symptoms: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pain_level": 3,
                "nausea": False,
                "dizziness": True,
                "fatigue": True,
                "custom_notes": "Feeling a bit tired today"
            }
        }
    )


class VitalSigns(BaseModel):
    """Vital signs and general health indicators"""
    mood: Optional[str] = None  # Removed strict pattern
    sleep_quality: Optional[str] = None  # Removed strict pattern
    appetite: Optional[str] = None  # Removed strict pattern
    energy_level: Optional[str] = None  # Removed strict pattern
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mood": "good",
                "sleep_quality": "fair",
                "appetite": "good",
                "energy_level": "medium"
            }
        }
    )


class AIInteraction(BaseModel):
    """AI conversation interaction data"""
    questions_asked: List[str] = Field(default_factory=list)
    responses_given: List[str] = Field(default_factory=list)
    conversation_summary: Optional[str] = None
    sentiment_analysis: Optional[str] = Field(None, pattern="^(positive|neutral|negative)$")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "questions_asked": [
                    "How are you feeling today?",
                    "Did you sleep well?"
                ],
                "responses_given": [
                    "I'm feeling okay",
                    "Yes, I slept well"
                ],
                "conversation_summary": "User reports feeling okay with good sleep",
                "sentiment_analysis": "positive"
            }
        }
    )


class HealthDataEntry(BaseModel):
    """Complete health data entry"""
    entry_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: str = "default_user"
    medication_id: Optional[UUID] = None
    symptoms: HealthSymptoms
    vital_signs: VitalSigns
    ai_interaction: Optional[AIInteraction] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user_001",
                "medication_id": "550e8400-e29b-41d4-a716-446655440000",
                "symptoms": {
                    "pain_level": 2,
                    "fatigue": True
                },
                "vital_signs": {
                    "mood": "good",
                    "sleep_quality": "good"
                }
            }
        }
    )


# ========== HARDWARE/DISPENSING MODELS ==========

class DispensingEvent(BaseModel):
    """Event log for medication dispensing"""
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    compartment: int = Field(..., ge=1, le=4)
    medication_id: UUID
    status: str = Field(..., pattern="^(success|failed|skipped)$")
    box_opened: bool
    led_activated: bool
    servo_response: Optional[str] = None
    error_message: Optional[str] = None
    conversation_session_id: Optional[UUID] = None
    ai_greeting: Optional[str] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "compartment": 1,
                "medication_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "success",
                "box_opened": True,
                "led_activated": True,
                "servo_response": "Servo opened successfully"
            }
        }
    )


class DispenseRequest(BaseModel):
    """Request to dispense medication"""
    compartment: int = Field(..., ge=1, le=4, description="Compartment to dispense from")
    medication_id: UUID
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "compartment": 1,
                "medication_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }
    )


class HardwareStatus(BaseModel):
    """Current hardware status"""
    servo: Dict[str, Any]
    leds: Dict[str, str]
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "servo": {
                    "position": "closed",
                    "operational": True,
                    "controller_reachable": True
                },
                "leds": {
                    "1": "off",
                    "2": "off",
                    "3": "off",
                    "4": "off"
                }
            }
        }
    )


# ========== AI CONVERSATION MODELS ==========

class ConversationMessage(BaseModel):
    """Single message in a conversation"""
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "role": "assistant",
                "content": "Hello! How are you feeling today?",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    )


class ConversationSession(BaseModel):
    """AI conversation session"""
    session_id: UUID = Field(default_factory=uuid4)
    medication_id: Optional[UUID] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    messages: List[ConversationMessage] = Field(default_factory=list)
    extracted_symptoms: Optional[Dict[str, Any]] = None
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "medication_id": "550e8400-e29b-41d4-a716-446655440000",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "Hello! How are you feeling?"
                    }
                ]
            }
        }
    )


class ConversationStartRequest(BaseModel):
    """Request to start a new conversation"""
    user_id: str
    medication_id: Optional[UUID] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user_001",
                "medication_id": None
            }
        }
    )


class MessageRequest(BaseModel):
    """Request to send a message in conversation"""
    message: str = Field(..., min_length=1, max_length=2000)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "I'm feeling okay, just a bit tired"
            }
        }
    )


# ========== VOICE INTERACTION MODELS ==========

class VoiceInteractionRequest(BaseModel):
    """Request for voice interaction"""
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio")
    text_input: Optional[str] = Field(None, description="Text alternative to audio")
    medication_id: Optional[UUID] = None
    language: str = Field("en", pattern="^[a-z]{2}$", description="ISO 639-1 language code")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text_input": "I have a headache",
                "medication_id": None,
                "language": "en"
            }
        }
    )


class VoiceInteractionResponse(BaseModel):
    """Response from voice interaction"""
    session_id: UUID
    ai_response_text: str
    ai_response_audio: Optional[str] = Field(None, description="Base64 encoded audio")
    extracted_symptoms: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None
    follow_up_questions: Optional[List[str]] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "ai_response_text": "I'm sorry to hear you have a headache. On a scale of 1 to 10, how would you rate the pain?",
                "extracted_symptoms": {
                    "headache": True,
                    "pain_level": None
                },
                "follow_up_questions": [
                    "How long have you had the headache?",
                    "Have you taken any medication for it?"
                ]
            }
        }
    )


# ========== SYMPTOM ANALYSIS MODELS ==========

class SymptomExtractionResult(BaseModel):
    """Result from symptom extraction"""
    symptoms: Dict[str, Any]
    confidence_scores: Dict[str, float] = {}
    extracted_entities: List[str] = []
    sentiment: str = "neutral"  # Ensure default value
    urgency_level: str = "low"
    extracted_count: int = 0

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symptoms": {
                    "pain_level": 5,
                    "headache": True,
                    "nausea": False
                },
                "confidence_scores": {
                    "pain_level": 0.9,
                    "headache": 0.95
                },
                "extracted_entities": ["headache", "pain"],
                "sentiment": "neutral",
                "urgency_level": "medium",
                "concerns": ["persistent headache"],
                "notes": "User reports moderate headache"
            }
        }
    )


# ========== RECOMMENDATION MODELS ==========

class HealthRecommendation(BaseModel):
    """Health recommendation generated by AI"""
    recommendation_id: UUID = Field(default_factory=uuid4)
    recommendation_text: str = Field(..., min_length=1)
    category: str = Field(
        ...,
        description="Category of recommendation",
        pattern="^(hydration|rest|activity|nutrition|seek_help|general)$"
    )
    priority: str = Field(..., pattern="^(low|medium|high)$")
    based_on: List[str] = Field(default_factory=list, description="Symptoms/data that led to this recommendation")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "recommendation_text": "Try drinking a glass of water and resting in a quiet, dark room for 20-30 minutes",
                "category": "rest",
                "priority": "medium",
                "based_on": ["headache", "fatigue"]
            }
        }
    )


# ========== REQUEST/RESPONSE MODELS ==========

class MarkTakenRequest(BaseModel):
    """Request to mark medication as taken"""
    medication_id: UUID
    timestamp: Optional[datetime] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "medication_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2024-01-15T08:05:00Z"
            }
        }
    )


class HealthLogRequest(BaseModel):
    """Request to create a health log entry"""
    medication_id: Optional[UUID] = None
    symptoms: HealthSymptoms
    vital_signs: VitalSigns
    notes: Optional[str] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symptoms": {
                    "pain_level": 2,
                    "fatigue": True
                },
                "vital_signs": {
                    "mood": "good",
                    "sleep_quality": "fair"
                },
                "notes": "Feeling better today"
            }
        }
    )


class DashboardData(BaseModel):
    """Compiled data for frontend dashboard"""
    user_id: str
    current_date: datetime = Field(default_factory=datetime.utcnow)
    hardware_status: HardwareStatus
    medications: Dict[str, Any]
    health_summary: Dict[str, Any]
    recent_interactions: List[Dict[str, Any]] = Field(default_factory=list)
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user_001",
                "medications": {
                    "today": [],
                    "next_dose": None,
                    "adherence_rate": 0.95
                },
                "health_summary": {
                    "period": "week",
                    "average_pain_level": 3.2,
                    "common_symptoms": ["fatigue"]
                },
                "alerts": [
                    {
                        "type": "medication_due",
                        "message": "Time to take your medication",
                        "priority": "high"
                    }
                ]
            }
        }
    )


# ========== ERROR RESPONSE MODEL ==========

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: bool = True
    error_code: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": True,
                "error_code": "HARDWARE_FAILURE",
                "message": "Failed to communicate with hardware controller",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    )


# ========== SUCCESS RESPONSE MODEL ==========

class SuccessResponse(BaseModel):
    """Standard success response"""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "data": {}
            }
        }
    )
