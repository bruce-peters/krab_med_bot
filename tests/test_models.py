"""
Test Pydantic Models
"""

import pytest
from datetime import datetime
from uuid import uuid4
from server.models.schemas import (
    MedicationEntry,
    HealthSymptoms,
    VitalSigns,
    AIInteraction,
    HealthDataEntry,
    DispensingEvent,
    DispenseRequest,
    ConversationMessage,
    ConversationSession,
    VoiceInteractionRequest,
    SymptomExtractionResult,
    HealthRecommendation,
)


def test_medication_entry_creation():
    """Test creating a medication entry"""
    med = MedicationEntry(
        name="Test Medication",
        compartment=1,
        scheduled_time=datetime.utcnow()
    )
    
    assert med.name == "Test Medication"
    assert med.compartment == 1
    assert med.taken == False
    assert med.medication_id is not None


def test_medication_entry_validation():
    """Test medication entry validation"""
    # Invalid compartment number
    with pytest.raises(ValueError):
        MedicationEntry(
            name="Test",
            compartment=5,  # Should be 1-4
            scheduled_time=datetime.utcnow()
        )


def test_health_symptoms():
    """Test health symptoms model"""
    symptoms = HealthSymptoms(
        pain_level=5,
        nausea=True,
        dizziness=False,
        custom_notes="Feeling unwell"
    )
    
    assert symptoms.pain_level == 5
    assert symptoms.nausea == True
    assert symptoms.dizziness == False


def test_vital_signs():
    """Test vital signs model"""
    vitals = VitalSigns(
        mood="good",
        sleep_quality="fair",
        appetite="good"
    )
    
    assert vitals.mood == "good"
    assert vitals.sleep_quality == "fair"


def test_ai_interaction():
    """Test AI interaction model"""
    interaction = AIInteraction(
        questions_asked=["How are you?", "Any pain?"],
        responses_given=["I'm okay", "A little"],
        conversation_summary="User reports minor discomfort"
    )
    
    assert len(interaction.questions_asked) == 2
    assert len(interaction.responses_given) == 2


def test_health_data_entry():
    """Test complete health data entry"""
    entry = HealthDataEntry(
        symptoms=HealthSymptoms(pain_level=3, fatigue=True),
        vital_signs=VitalSigns(mood="okay", sleep_quality="good"),
        medication_id=uuid4()
    )
    
    assert entry.symptoms.pain_level == 3
    assert entry.vital_signs.mood == "okay"
    assert entry.entry_id is not None


def test_dispensing_event():
    """Test dispensing event model"""
    event = DispensingEvent(
        compartment=2,
        medication_id=uuid4(),
        status="success",
        box_opened=True,
        led_activated=True
    )
    
    assert event.compartment == 2
    assert event.status == "success"
    assert event.event_id is not None


def test_conversation_message():
    """Test conversation message model"""
    msg = ConversationMessage(
        role="assistant",
        content="Hello! How are you feeling today?"
    )
    
    assert msg.role == "assistant"
    assert msg.content is not None
    assert msg.timestamp is not None


def test_conversation_session():
    """Test conversation session model"""
    session = ConversationSession(
        user_id="test_user",
        medication_id=uuid4()
    )
    
    assert session.user_id == "test_user"
    assert session.session_id is not None
    assert len(session.messages) == 0


def test_symptom_extraction_result():
    """Test symptom extraction result"""
    result = SymptomExtractionResult(
        symptoms={"headache": True, "pain_level": 4},
        confidence_scores={"headache": 0.95},
        extracted_entities=["headache", "pain"],
        sentiment="neutral",
        urgency_level="medium"
    )
    
    assert result.urgency_level == "medium"
    assert result.sentiment == "neutral"
    assert "headache" in result.extracted_entities


def test_health_recommendation():
    """Test health recommendation model"""
    rec = HealthRecommendation(
        recommendation_text="Drink more water",
        category="hydration",
        priority="medium",
        based_on=["fatigue", "headache"]
    )
    
    assert rec.category == "hydration"
    assert rec.priority == "medium"
    assert len(rec.based_on) == 2


def test_voice_interaction_request():
    """Test voice interaction request"""
    req = VoiceInteractionRequest(
        text_input="I have a headache",
        language="en"
    )
    
    assert req.text_input == "I have a headache"
    assert req.language == "en"
    assert req.audio_data is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
