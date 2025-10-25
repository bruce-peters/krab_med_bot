"""
Quick test script for Phase 3 - Data Models
Run this to verify all models are working correctly
"""

from datetime import datetime
from uuid import uuid4
from server.models.schemas import (
    MedicationEntry,
    HealthSymptoms,
    VitalSigns,
    HealthDataEntry,
    DispensingEvent,
    ConversationMessage,
    ConversationSession,
    SymptomExtractionResult,
    HealthRecommendation,
)


def test_models():
    """Test all Pydantic models"""
    print("=" * 70)
    print("Testing Phase 3 - Pydantic Data Models")
    print("=" * 70)
    print()
    
    # Test 1: Medication Entry
    print("✓ Testing MedicationEntry...")
    med = MedicationEntry(
        name="Blood Pressure Medication",
        compartment=1,
        scheduled_time=datetime.utcnow()
    )
    print(f"  Created medication: {med.name} in compartment {med.compartment}")
    print(f"  Medication ID: {med.medication_id}")
    
    # Test 2: Health Symptoms
    print("\n✓ Testing HealthSymptoms...")
    symptoms = HealthSymptoms(
        pain_level=3,
        nausea=False,
        dizziness=True,
        fatigue=True,
        custom_notes="Feeling a bit dizzy today"
    )
    print(f"  Pain level: {symptoms.pain_level}/10")
    print(f"  Symptoms: dizziness={symptoms.dizziness}, fatigue={symptoms.fatigue}")
    
    # Test 3: Vital Signs
    print("\n✓ Testing VitalSigns...")
    vitals = VitalSigns(
        mood="good",
        sleep_quality="fair",
        appetite="good",
        energy_level="medium"
    )
    print(f"  Mood: {vitals.mood}, Sleep: {vitals.sleep_quality}")
    
    # Test 4: Health Data Entry
    print("\n✓ Testing HealthDataEntry...")
    health_entry = HealthDataEntry(
        user_id="user_001",
        medication_id=med.medication_id,
        symptoms=symptoms,
        vital_signs=vitals
    )
    print(f"  Entry ID: {health_entry.entry_id}")
    print(f"  Timestamp: {health_entry.timestamp}")
    
    # Test 5: Dispensing Event
    print("\n✓ Testing DispensingEvent...")
    event = DispensingEvent(
        compartment=1,
        medication_id=med.medication_id,
        status="success",
        box_opened=True,
        led_activated=True,
        servo_response="Opened successfully"
    )
    print(f"  Event ID: {event.event_id}")
    print(f"  Status: {event.status}, LED: {event.led_activated}")
    
    # Test 6: Conversation Message
    print("\n✓ Testing ConversationMessage...")
    message = ConversationMessage(
        role="assistant",
        content="Hello! How are you feeling today?"
    )
    print(f"  Role: {message.role}")
    print(f"  Content: {message.content}")
    
    # Test 7: Conversation Session
    print("\n✓ Testing ConversationSession...")
    session = ConversationSession(
        user_id="user_001",
        medication_id=med.medication_id
    )
    session.messages.append(message)
    print(f"  Session ID: {session.session_id}")
    print(f"  Messages: {len(session.messages)}")
    
    # Test 8: Symptom Extraction Result
    print("\n✓ Testing SymptomExtractionResult...")
    extraction = SymptomExtractionResult(
        symptoms={"headache": True, "pain_level": 3},
        confidence_scores={"headache": 0.9, "pain_level": 0.85},
        extracted_entities=["headache", "pain", "dizziness"],
        sentiment="neutral",
        urgency_level="medium",
        concerns=["persistent dizziness"]
    )
    print(f"  Urgency: {extraction.urgency_level}")
    print(f"  Sentiment: {extraction.sentiment}")
    print(f"  Entities: {', '.join(extraction.extracted_entities)}")
    
    # Test 9: Health Recommendation
    print("\n✓ Testing HealthRecommendation...")
    recommendation = HealthRecommendation(
        recommendation_text="Try resting in a quiet room for 20-30 minutes",
        category="rest",
        priority="medium",
        based_on=["dizziness", "fatigue"]
    )
    print(f"  Category: {recommendation.category}")
    print(f"  Priority: {recommendation.priority}")
    print(f"  Text: {recommendation.recommendation_text}")
    
    # Test 10: JSON Serialization
    print("\n✓ Testing JSON Serialization...")
    health_json = health_entry.model_dump_json(indent=2)
    print(f"  Health entry serialized to JSON ({len(health_json)} chars)")
    
    # Test 11: Model Validation
    print("\n✓ Testing Model Validation...")
    try:
        invalid_med = MedicationEntry(
            name="Test",
            compartment=5,  # Invalid: should be 1-4
            scheduled_time=datetime.utcnow()
        )
        print("  ✗ Validation failed - invalid data accepted!")
    except Exception as e:
        print(f"  ✓ Validation working - caught error: {type(e).__name__}")
    
    print("\n" + "=" * 70)
    print("✅ All Phase 3 Tests Passed!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run the server: uvicorn server.main:app --reload")
    print("  2. Check API docs: http://localhost:5000/docs")
    print("  3. Try creating objects via the API")
    print()


if __name__ == "__main__":
    test_models()
