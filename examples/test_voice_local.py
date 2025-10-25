"""
Test script for voice interaction with local audio playback
Run this to test the voice features without handling audio data
"""

import requests
import time
from pathlib import Path

BASE_URL = "http://localhost:5000"

def test_voice_conversation_text():
    """Test a complete voice conversation with text input"""
    
    print("🎤 Testing Voice Interaction with Text Input")
    print("=" * 60)
    
    # Start conversation
    print("\n1. Starting conversation...")
    response = requests.post(
        f"{BASE_URL}/api/voice/interact/test",
        data={"text_input": "Hello, I just took my medication"}
    )
    data = response.json()
    session_id = data['session_id']
    
    print(f"   Session ID: {session_id}")
    print(f"   AI Response: {data['ai_response_text']}")
    print(f"   🔊 Audio played locally: {data['audio_played_locally']}")
    
    time.sleep(2)
    
    # Continue conversation
    messages = [
        "I'm feeling a bit dizzy",
        "It started this morning",
        "My pain level is about 3 out of 10"
    ]
    
    for i, message in enumerate(messages, 2):
        print(f"\n{i}. Sending message: '{message}'")
        response = requests.post(
            f"{BASE_URL}/api/voice/interact/test",
            data={
                "session_id": session_id,
                "text_input": message
            }
        )
        data = response.json()
        
        print(f"   AI Response: {data['ai_response_text']}")
        print(f"   🔊 Audio played locally")
        
        if data.get('extracted_symptoms'):
            print(f"   📋 Detected symptoms: {data['extracted_symptoms']}")
        
        time.sleep(2)
    
    # End conversation
    print(f"\n{len(messages) + 2}. Ending conversation...")
    response = requests.post(
        f"{BASE_URL}/api/voice/conversation/{session_id}/end/test"
    )
    data = response.json()
    
    print(f"   Urgency Level: {data['urgency_level']}")
    print(f"   Closing: {data['closing_message']}")
    print(f"   🔊 Audio played locally")
    print(f"\n   Recommendations:")
    for rec in data['recommendations']:
        print(f"   - {rec['recommendation_text']}")
    
    print("\n" + "=" * 60)
    print("✅ Test completed successfully!")

def test_voice_conversation_audio():
    """Test voice conversation with audio file upload"""
    
    print("\n🎙️ Testing Voice Interaction with Audio File")
    print("=" * 60)
    
    # Check if test audio file exists
    audio_file_path = Path("test_audio.wav")
    if not audio_file_path.exists():
        print("⚠️  No test audio file found (test_audio.wav)")
        print("   Create a WAV file or skip this test")
        return
    
    # Start conversation with audio
    print("\n1. Starting conversation with audio file...")
    with open(audio_file_path, 'rb') as audio_file:
        response = requests.post(
            f"{BASE_URL}/api/voice/interact/test",
            files={"audio_file": audio_file}
        )
    data = response.json()
    session_id = data['session_id']
    
    print(f"   Session ID: {session_id}")
