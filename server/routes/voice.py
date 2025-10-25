from fastapi import APIRouter, UploadFile, File, HTTPException
from server.models.schemas import VoiceInteractionRequest, VoiceInteractionResponse
from server.ai.speech import speech_service
from server.ai.conversation import conversation_manager
from server.ai.symptom_analyzer import symptom_analyzer
from server.ai.recommendation import recommendation_engine
from server.config import settings
import base64
import logging
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice", tags=["voice"])


@router.post("/interact", response_model=VoiceInteractionResponse)
async def voice_interaction(request: VoiceInteractionRequest):
    """
    Handle voice interaction with AI
    - Accepts audio or text input
    - Returns AI response as audio and text
    - Extracts symptoms automatically
    - Generates personalized recommendations
    """
    try:
        # Convert audio to text if provided
        if request.audio_data:
            logger.info("Processing audio input for transcription")
            user_text = await speech_service.speech_to_text(
                request.audio_data,
                language=request.language
            )
            logger.info(f"Transcribed text: {user_text}")
        elif request.text_input:
            user_text = request.text_input
            logger.info(f"Processing text input: {user_text}")
        else:
            raise HTTPException(
                status_code=400,
                detail="Either audio_data or text_input must be provided"
            )

        # Start or continue conversation
        # For simplicity, create new session each time (could track session)
        session = await conversation_manager.start_session(
            "voice_user",
            request.medication_id
        )
        logger.info(f"Started conversation session: {session.session_id}")

        # Generate AI response
        ai_text = await conversation_manager.generate_response(
            session.session_id,
            user_text
        )
        logger.info(f"Generated AI response: {ai_text}")

        # Convert response to speech
        ai_audio = None
        if settings.enable_voice_interaction:
            try:
                ai_audio = await speech_service.text_to_speech(
                    ai_text,
                    voice=settings.tts_voice
                )
                logger.info("Generated audio response")
            except Exception as e:
                logger.error(f"TTS conversion failed: {e}")
                # Continue without audio - still return text response

        # Extract symptoms if auto-extraction enabled
        extracted_symptoms = None
        recommendations = None
        if settings.auto_symptom_extraction:
            logger.info("Extracting symptoms from conversation")
            conversation_text = f"User: {user_text}\nAI: {ai_text}"
            symptom_result = await symptom_analyzer.extract_symptoms(conversation_text)
            extracted_symptoms = symptom_result.symptoms

            if extracted_symptoms:
                logger.info(f"Extracted symptoms: {extracted_symptoms}")
                # Generate recommendations based on symptoms
                recommendations_list = await recommendation_engine.generate_recommendations(
                    extracted_symptoms,
                    {}  # TODO: Get user history from database
                )
                recommendations = [r.recommendation_text for r in recommendations_list]
                logger.info(f"Generated {len(recommendations)} recommendations")

        # Generate contextual follow-up questions based on conversation
        follow_ups = _generate_follow_up_questions(user_text, extracted_symptoms)

        return VoiceInteractionResponse(
            session_id=session.session_id,
            ai_response_text=ai_text,
            ai_response_audio=ai_audio,
            extracted_symptoms=extracted_symptoms,
            recommendations=recommendations,
            follow_up_questions=follow_ups[:2]  # Limit to 2
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice interaction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Voice interaction failed: {str(e)}"
        )


@router.post("/upload-audio")
async def upload_audio_file(
    file: UploadFile = File(...),
    language: str = "en"
):
    """
    Upload audio file for transcription
    Alternative to base64 encoding for larger files
    Supports: WAV, MP3, M4A, WEBM
    """
    try:
        # Validate file type
        allowed_types = ["audio/wav", "audio/mpeg", "audio/mp4", "audio/webm"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format. Allowed: {allowed_types}"
            )

        logger.info(f"Uploading audio file: {file.filename}")

        # Read audio bytes
        audio_bytes = await file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Transcribe
        transcription = await speech_service.speech_to_text(
            audio_base64,
            language=language
        )

        logger.info(f"Transcription successful: {transcription}")

        return {
            "transcription": transcription,
            "filename": file.filename,
            "language": language
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio upload error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Audio upload failed: {str(e)}"
        )


@router.post("/text-to-speech")
async def convert_text_to_speech(
    text: str,
    voice: Optional[str] = None
):
    """
    Convert text to speech audio
    Returns base64 encoded audio
    """
    try:
        if not text or len(text.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty"
            )

        voice_to_use = voice or settings.tts_voice

        logger.info(f"Converting text to speech with voice: {voice_to_use}")

        audio_base64 = await speech_service.text_to_speech(
            text,
            voice=voice_to_use
        )

        return {
            "audio_data": audio_base64,
            "text": text,
            "voice": voice_to_use
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS conversion error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"TTS conversion failed: {str(e)}"
        )


@router.get("/voices")
async def list_available_voices():
    """
    List available TTS voices
    """
    # OpenAI TTS voices
    openai_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    return {
        "provider": settings.tts_provider,
        "voices": openai_voices if settings.tts_provider == "openai" else [],
        "current_voice": settings.tts_voice
    }


def _generate_follow_up_questions(
    user_text: str,
    symptoms: Optional[dict]
) -> list[str]:
    """
    Generate contextual follow-up questions based on conversation
    """
    user_text_lower = user_text.lower()
    questions = []

    # Pain-related follow-ups
    if any(word in user_text_lower for word in ["pain", "hurt", "ache"]):
        if not symptoms or "pain_level" not in symptoms:
            questions.append("How would you rate your pain on a scale of 1 to 10?")
        questions.append("Where exactly does it hurt?")

    # Sleep-related follow-ups
    if any(word in user_text_lower for word in ["tired", "sleep", "rest"]):
        questions.append("Did you sleep well last night?")
        questions.append("How many hours of sleep did you get?")

    # Digestive-related follow-ups
    if any(word in user_text_lower for word in ["nausea", "stomach", "appetite"]):
        questions.append("How is your appetite today?")
        questions.append("Have you been able to eat normally?")

    # Mood-related follow-ups
    if any(word in user_text_lower for word in ["sad", "anxious", "worried", "stressed"]):
        questions.append("Is there anything specific that's bothering you?")
        questions.append("Would you like to talk about how you're feeling?")

    # General follow-ups if no specific topics detected
    if not questions:
        questions = [
            "How are you feeling overall today?",
            "Are you experiencing any discomfort?",
            "Is there anything else you'd like to share?"
        ]

    return questions

