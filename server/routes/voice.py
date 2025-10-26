"""
Voice Interaction Routes
Handles voice-based conversations with speech-to-text and text-to-speech
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from server.models.schemas import VoiceInteractionRequest, VoiceInteractionResponse
from server.ai.conversation import conversation_manager
from server.ai.speech import speech_service
from uuid import UUID
from typing import Optional
import logging
import base64

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/voice", tags=["voice"])

@router.post("/interact", response_model=VoiceInteractionResponse)
async def voice_interact(request: Request):
    """
    Complete voice interaction cycle
    
    Flow:
    1. Convert speech to text (if audio provided)
    2. Get AI response from conversation
    3. Convert response to speech
    4. Return text + audio response
    
    Usage:
        # Start new conversation
        POST /api/voice/interact
        {
            "text_input": "I'm feeling okay",
            "medication_id": "uuid-or-null"
        }
        
        # Continue conversation
        POST /api/voice/interact
        {
            "session_id": "session-uuid",
            "text_input": "I have a headache"
        }
    """
    logger.info("Received request for /api/voice/interact")
    
    # Log raw request body
    raw_body = await request.body()
    logger.debug(f"Raw request body: {raw_body.decode('utf-8')}")
    
    # Parse request
    try:
        data = await request.json()
        logger.debug(f"Parsed request data: {data}")
    except Exception as e:
        logger.error(f"Failed to parse request: {e}")
        raise HTTPException(status_code=400, detail="Invalid request format")
    
    session_id = data.get("session_id")
    audio_data = data.get("audio_data")
    text_input = data.get("text_input")
    medication_id = data.get("medication_id")
    
    logger.debug(f"Parsed session_id: {session_id}")
    logger.info(f"Request data - session_id: {session_id}, medication_id: {medication_id}, text_input: {text_input}, audio_data: {'provided' if audio_data else 'not provided'}")
    
    try:
        # Convert session_id and medication_id from string to UUID if provided
        session_uuid = UUID(session_id) if session_id else None
        medication_uuid = UUID(medication_id) if medication_id else None
        
        # Start or get existing session
        if session_uuid is None:
            logger.info("No session_id provided. Starting a new session.")
            session = await conversation_manager.start_session(medication_uuid)
            session_uuid = session.session_id
            
            # Generate greeting
            greeting = await conversation_manager.generate_response(
                session_uuid,
                "User has started voice interaction."
            )
            logger.debug(f"Generated greeting: {greeting}")
            
            # Convert greeting to speech
            greeting_audio = await speech_service.text_to_speech(greeting)
            logger.debug("Converted greeting to audio.")
            
            return VoiceInteractionResponse(
                session_id=session_uuid,
                ai_response_text=greeting,
                ai_response_audio=greeting_audio,
                extracted_symptoms=None,
                recommendations=None,
                follow_up_questions=_generate_follow_ups(greeting)
            )
        
        # Process user input
        user_text = text_input
        if audio_data and not user_text:
            # Convert speech to text
            logger.info("Converting speech to text...")
            user_text = await speech_service.speech_to_text(audio_data)
            logger.info(f"Transcribed: {user_text[:100]}...")
        
        if not user_text:
            raise HTTPException(
                status_code=400,
                detail="Either audio_data or text_input must be provided"
            )
        
        # Get AI response
        ai_response = await conversation_manager.generate_response(session_uuid, user_text)

        # Extract follow-up questions from the response
        follow_up_questions = _generate_follow_ups(ai_response)
        
        # Convert response to speech
        response_audio = await speech_service.text_to_speech(ai_response)
        
        # Get current session for symptom extraction (lightweight check)
        session = conversation_manager.active_sessions.get(session_uuid)
        current_symptoms = None
        if session and len(session.messages) > 4:  # After a few exchanges
            from server.ai.symptom_analyzer import symptom_analyzer
            try:
                conversation_text = "\n".join([
                    f"{msg.role}: {msg.content}"
                    for msg in session.messages[-6:]  # Last 6 messages
                    if msg.role != "system"
                ])
                symptoms = await symptom_analyzer.extract_symptoms(conversation_text)
                if symptoms.symptoms:
                    current_symptoms = symptoms.symptoms
            except Exception as e:
                logger.warning(f"Lightweight symptom extraction failed: {e}")
        
        return VoiceInteractionResponse(
            session_id=session_uuid,
            ai_response_text=ai_response,
            ai_response_audio=response_audio,
            extracted_symptoms=current_symptoms,
            recommendations=None,  # Only provided at end of conversation
            follow_up_questions=follow_up_questions
        )
        
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation session not found")
    except Exception as e:
        logger.error(f"Error in voice interaction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Voice interaction failed: {str(e)}")

@router.post("/interact/file")
async def voice_interact_file(
    session_id: Optional[str] = Form(None),
    medication_id: Optional[str] = Form(None),
    audio_file: UploadFile = File(...)
):
    """
    Voice interaction with audio file upload
    
    Usage:
        curl -X POST http://localhost:5000/api/voice/interact/file \
          -F "audio_file=@recording.wav" \
          -F "session_id=uuid-here"
    """
    try:
        # Read audio file and convert to base64
        audio_bytes = await audio_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Use main interact endpoint logic
        return await voice_interact(
            session_id=session_id,
            audio_data=audio_base64,
            text_input=None,
            medication_id=medication_id
        )
        
    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

@router.post("/interact/test")
async def voice_interact_test(
    session_id: Optional[str] = Form(None),
    text_input: str = Form(...),
    medication_id: Optional[str] = Form(None),
    play_audio: bool = Form(True)
):
    """
    Test voice interaction with local audio playback
    
    This endpoint is for testing - it plays the AI response audio locally
    on the server instead of returning base64.
    
    Usage:
        curl -X POST http://localhost:5000/api/voice/interact/test \
          -F "text_input=I have a headache" \
          -F "play_audio=true"
    """
    try:
        # Convert session_id and medication_id from string to UUID if provided
        session_uuid = UUID(session_id) if session_id else None
        medication_uuid = UUID(medication_id) if medication_id else None
        
        # Start or get existing session
        if session_uuid is None:
            session = await conversation_manager.start_session(medication_uuid)
            session_uuid = session.session_id
            
            # Generate greeting
            greeting = await conversation_manager.generate_response(
                session_uuid,
                "User has started voice interaction."
            )
            
            # Play audio locally if requested
            if play_audio:
                greeting_audio = await speech_service.text_to_speech(greeting)
                await _play_audio_locally(greeting_audio)
            
            return {
                "session_id": str(session_uuid),
                "ai_response_text": greeting,
                "audio_played_locally": play_audio,
                "follow_up_questions": _generate_follow_ups(greeting)
            }
        
        # Get AI response
        ai_response = await conversation_manager.generate_response(session_uuid, text_input)
        
        # Play audio locally if requested
        if play_audio:
            response_audio = await speech_service.text_to_speech(ai_response)
            await _play_audio_locally(response_audio)
        
        # Get current session for symptom extraction
        session = conversation_manager.active_sessions.get(session_uuid)
        current_symptoms = None
        if session and len(session.messages) > 4:
            from server.ai.symptom_analyzer import symptom_analyzer
            try:
                conversation_text = "\n".join([
                    f"{msg.role}: {msg.content}"
                    for msg in session.messages[-6:]
                    if msg.role != "system"
                ])
                symptoms = await symptom_analyzer.extract_symptoms(conversation_text)
                if symptoms.symptoms:
                    current_symptoms = symptoms.symptoms
            except Exception as e:
                logger.warning(f"Lightweight symptom extraction failed: {e}")
        
        return {
            "session_id": str(session_uuid),
            "user_input": text_input,
            "ai_response_text": ai_response,
            "audio_played_locally": play_audio,
            "extracted_symptoms": current_symptoms,
            "follow_up_questions": _generate_follow_ups(ai_response)
        }
        
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation session not found")
    except Exception as e:
        logger.error(f"Error in test voice interaction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Voice interaction failed: {str(e)}")

@router.post("/conversation/{session_id}/end")
async def end_voice_conversation(session_id: UUID):
    """
    End voice conversation with analysis
    
    Uses centralized conversation ending logic.
    Returns analysis with TTS audio for closing message and recommendations.
    
    Usage:
        POST /api/voice/conversation/{session_id}/end
    """
    try:
        # Use centralized conversation ending logic
        result = await conversation_manager.end_session_with_analysis(session_id)
        
        # Generate closing message based on urgency
        closing_message = _generate_closing_message(
            result["urgency_level"],
            result["symptoms"].sentiment
        )
        
        # Convert closing message to speech
        closing_audio = await speech_service.text_to_speech(closing_message)
        
        # Convert recommendations to speech (first 3 only)
        recommendation_audio = []
        for rec in result["recommendations"][:3]:
            audio = await speech_service.text_to_speech(rec.recommendation_text)
            recommendation_audio.append({
                "text": rec.recommendation_text,
                "audio": audio,
                "category": rec.category
            })
        
        return {
            **result,
            "closing_message": closing_message,
            "closing_audio": closing_audio,
            "recommendation_audio": recommendation_audio
        }
        
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation session not found")
    except Exception as e:
        logger.error(f"Error ending voice conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to end voice conversation: {str(e)}")

@router.post("/conversation/{session_id}/end/test")
async def end_voice_conversation_test(
    session_id: UUID,
    play_audio: bool = True
):
    """
    End voice conversation with local audio playback (for testing)
    
    Plays the closing message and recommendations locally instead of returning base64.
    """
    try:
        # Use centralized conversation ending logic
        result = await conversation_manager.end_session_with_analysis(session_id)
        
        # Generate closing message
        closing_message = _generate_closing_message(
            result["urgency_level"],
            result["symptoms"].sentiment
        )
        
        # Play closing message locally if requested
        if play_audio:
            closing_audio = await speech_service.text_to_speech(closing_message)
            await _play_audio_locally(closing_audio)
            
            # Also play first recommendation
            if result["recommendations"]:
                rec_text = f"Recommendation: {result['recommendations'][0].recommendation_text}"
                rec_audio = await speech_service.text_to_speech(rec_text)
                await _play_audio_locally(rec_audio)
        
        return {
            "session_id": session_id,
            "symptoms": result["symptoms"],
            "recommendations": result["recommendations"],
            "urgency_level": result["urgency_level"],
            "closing_message": closing_message,
            "audio_played_locally": play_audio,
            "conversation_summary": result["conversation_summary"]
        }
        
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation session not found")
    except Exception as e:
        logger.error(f"Error ending test voice conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to end voice conversation: {str(e)}")

@router.get("/session/{session_id}/status")
async def get_voice_session_status(session_id: UUID):
    """
    Get current status of voice session
    
    Returns conversation state and message count
    """
    try:
        session = conversation_manager.active_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "started_at": session.started_at,
            "message_count": len(session.messages),
            "user_message_count": len([m for m in session.messages if m.role == "user"]),
            "is_active": session.ended_at is None,
            "medication_id": session.medication_id
        }
        
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _generate_follow_ups(ai_response: str) -> list:
    """Generate contextual follow-up questions based on AI response"""
    response_lower = ai_response.lower()
    
    follow_ups = []
    
    if "pain" in response_lower:
        follow_ups.append("Where exactly is the pain located?")
        follow_ups.append("Is the pain constant or does it come and go?")
    
    if "sleep" in response_lower:
        follow_ups.append("What time did you go to bed?")
        follow_ups.append("Did anything wake you up during the night?")
    
    if "feeling" in response_lower or "how are" in response_lower:
        follow_ups.append("Have you noticed any changes since yesterday?")
        follow_ups.append("Is there anything bothering you today?")
    
    # Default follow-ups
    if not follow_ups:
        follow_ups = [
            "Is there anything else you'd like to share?",
            "How is your energy level today?"
        ]
    
    return follow_ups[:2]  # Return max 2 follow-ups

def _generate_closing_message(urgency_level: str, sentiment: str) -> str:
    """Generate appropriate closing message based on analysis"""
    if urgency_level == "urgent":
        return "I'm concerned about what you've shared. Please contact your doctor right away or have someone call for help. Take care."
    
    elif urgency_level == "high":
        return "Thank you for sharing. I recommend calling your doctor today to discuss how you're feeling. Take it easy and rest."
    
    elif urgency_level == "medium":
        return "Thank you for the conversation. Please follow the suggestions I provided and don't hesitate to call your doctor if you need to."
    
    else:  # low
        if sentiment == "positive":
            return "It's wonderful to hear you're doing well! Keep up the good work with your medications. Have a great day!"
        else:
            return "Thank you for sharing. Remember to take your medication as prescribed. Take care and have a good day!"

async def _play_audio_locally(audio_base64: str):
    """
    Play audio locally on server using pygame (for testing)
    """
    import tempfile
    import base64
    import os
    import pygame
    import io

    # Decode base64 to bytes
    audio_bytes = base64.b64decode(audio_base64)

    pygame.mixer.init()
    sound_file = io.BytesIO(audio_bytes)  # Fixed variable name
    sound = pygame.mixer.Sound(sound_file)
    sound.play()