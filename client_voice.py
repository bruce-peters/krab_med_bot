"""
Continuous Voice Interaction Client for Krab Med Bot

This script continuously listens for user speech, sends it to the
backend API, and plays the audio response.
"""

import requests
import base64
import pygame
import speech_recognition as sr
import io
import time
import logging
from typing import Optional
import httpx  # Replace requests with httpx

# Configuration
SERVER_URL = "http://localhost:5000/api/voice/interact"
SESSION_ID: Optional[str] = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def listen_for_speech() -> Optional[bytes]:
    """Listen for user speech and return audio data"""
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        logger.info("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            logger.debug("Speech detected, processing audio...")
            return audio.get_wav_data()
        except sr.WaitTimeoutError:
            logger.warning("No speech detected")
            return None
        except Exception as e:
            logger.error(f"Error during speech recognition: {e}")
            return None


def send_to_server(audio_data: bytes) -> Optional[dict]:
    """Send audio to server and get response"""
    global SESSION_ID
    logger.debug(f"Session ID before request: {SESSION_ID}")
    
    # Encode audio as Base64
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    
    # Prepare form-data payload
    payload = {
        "session_id": SESSION_ID or "",
        "audio_data": audio_base64,
    }
    logger.debug(f"Payload being sent as form-data: {payload}")
    
    try:
        # Send request using requests
        response = requests.post(SERVER_URL, json=payload)  # Use `data` for form-data
        logger.debug(f"Server response status: {response.status_code}")
        logger.debug(f"Server response body: {response.text}")
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        logger.debug(f"Parsed server response: {result}")
        
        # Update session ID only if it's present in the response
        new_session_id = result.get("session_id")
        if new_session_id:
            logger.info(f"Updating session ID: {new_session_id}")
            SESSION_ID = new_session_id
        
        return result
        
    except requests.RequestException as e:
        logger.error(f"❌ HTTP request error: {e}")
        return None


def play_audio(audio_base64: str):
    """Play audio from Base64-encoded data"""
    try:
        # Decode Base64
        audio_bytes = base64.b64decode(audio_base64)
        
        # Initialize pygame
        pygame.mixer.init()
        
        # Create and play sound
        sound = pygame.mixer.Sound(io.BytesIO(audio_bytes))
        sound.play()
        
        # Wait for playback to finish
        while pygame.mixer.get_busy():
            pygame.time.wait(100)
            
        logger.info("🔊 Audio playback complete")
        
    except Exception as e:
        logger.error(f"❌ Audio playback error: {e}")


def main():
    """Main loop for continuous voice interaction"""
    logger.info("=" * 60)
    logger.info("Krab Med Bot - Voice Interaction Client")
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to exit")
    logger.info("")
    
    try:
        while True:
            # Listen for speech
            audio_data = listen_for_speech()
            
            if not audio_data:
                print("NO AUDIO DATA")
                continue
            
            logger.info("📤 Sending audio to server...")
            
            # Send to server
            response = send_to_server(audio_data)
            
            if not response:
                logger.error("Failed to get response from server")
                time.sleep(2)
                continue
            
            # Display text response
            ai_text = response.get("ai_response_text", "")
            logger.info(f"🤖 AI: {ai_text}")
            
            # Play audio response
            audio_response = response.get("ai_response_audio")
            if audio_response:
                play_audio(audio_response)
            
            # print(response)
            # Display symptoms if extracted
            symptoms = response.get("extracted_symptoms")
            if symptoms:
                logger.info(f"📋 Symptoms detected: {list(symptoms.keys())}")
            
            # Small pause before next listen
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        logger.info("\n👋 Ending conversation...")
        
        # End session if active
        if SESSION_ID:
            try:
                end_url = f"http://localhost:5000/api/voice/conversation/{SESSION_ID}/end"
                response = requests.post(end_url)
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"📊 Conversation summary: {result.get('conversation_summary')}")
            except Exception as e:
                logger.error(f"Error ending session: {e}")
        
        logger.info("Goodbye!")


if __name__ == "__main__":
    main()
