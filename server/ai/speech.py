"""
Speech Services - Text-to-Speech and Speech-to-Text
Supports multiple providers: OpenAI, Gemini, local pyttsx3, mock
"""

import logging
import base64
import io
import asyncio
import threading
from typing import Optional
from server.config import settings

logger = logging.getLogger(__name__)

# Global lock for pyttsx3 engine (only one instance can run at a time)
_tts_lock = threading.Lock()
_tts_engine = None

def _get_tts_engine():
    """Get or create the global pyttsx3 engine"""
    global _tts_engine
    if (_tts_engine is None):
        import pyttsx3
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty('rate', 150)
        _tts_engine.setProperty('volume', 0.9)
    return _tts_engine

def _synthesize_speech(engine, text, output_path):
    """
    Simplified method to synthesize speech without threading or locks.
    
    Args:
        engine: pyttsx3 engine instance
        text: Text to speak
        output_path: Path to save audio file
    """
    try:
        # Save to file
        engine.save_to_file(text, output_path)
        
        # Run the synthesis
        engine.runAndWait()
        logger.debug(f"TTS synthesis completed for: {text[:50]}...")
        
    except Exception as e:
        logger.error(f"pyttsx3 synthesis error: {e}")
        raise

class SpeechService:
    def __init__(self):
        self.stt_provider = settings.stt_provider
        self.tts_provider = settings.tts_provider
        self.tts_voice = settings.tts_voice
        
        # Initialize clients based on providers
        if self.stt_provider == "openai" or self.tts_provider == "openai":
            import openai
            self.openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        
        if self.stt_provider == "gemini" or self.tts_provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=settings.gemini_api_key)
            # Use Gemini 2.0 Flash for audio
            self.gemini_client = genai.GenerativeModel("gemini-2.0-flash-exp")
    
    async def speech_to_text(self, audio_base64: str) -> str:
        """
        Convert speech audio to text
        
        Args:
            audio_base64: Base64 encoded audio data
            
        Returns:
            Transcribed text
        """
        try:
            if self.stt_provider == "gemini":
                return await self._stt_gemini(audio_base64)
            elif self.stt_provider == "openai":
                return await self._stt_openai(audio_base64)
            elif self.stt_provider == "whisper_local":
                return await self._stt_whisper(audio_base64)
            else:  # mock
                return await self._stt_mock(audio_base64)
        except Exception as e:
            logger.error(f"STT error: {e}", exc_info=True)
            return "[Speech recognition failed]"
    
    async def text_to_speech(self, text: str) -> str:
        """
        Convert text to speech audio
        
        Args:
            text: Text to convert
            
        Returns:
            Base64 encoded audio data
        """
        try:
            if self.tts_provider == "local":
                return await self._tts_local(text)
            elif self.tts_provider == "openai":
                return await self._tts_openai(text)
            elif self.tts_provider == "elevenlabs":
                return await self._tts_elevenlabs(text)
            else:  # mock
                return await self._tts_mock(text)
        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            return ""  # Return empty audio on error
    
    async def _stt_gemini(self, audio_base64: str) -> str:
        """
        Gemini multimodal audio transcription
        
        Gemini 2.0 Flash supports direct audio input for transcription
        """
        try:
            # Decode base64 to bytes
            audio_bytes = base64.b64decode(audio_base64)
            
            # Upload audio file to Gemini
            import google.generativeai as genai
            
            # Save temporarily (Gemini API requires file upload)
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            try:
                # Upload file
                audio_file = genai.upload_file(temp_path)
                
                # Generate transcription
                response = await self.gemini_client.generate_content_async([
                    "Transcribe this audio to text. Return only the spoken words, nothing else.",
                    audio_file
                ])
                
                # Delete the uploaded file
                audio_file.delete()
                
                return response.text.strip()
                
            finally:
                # Clean up temp file
                import os
                os.unlink(temp_path)
                
        except Exception as e:
            logger.error(f"Gemini STT error: {e}", exc_info=True)
            # Fallback to mock
            return await self._stt_mock(audio_base64)
    
    async def _stt_openai(self, audio_base64: str) -> str:
        """OpenAI Whisper STT"""
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(audio_base64)
        
        # Create temp file (OpenAI requires file-like object)
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav"
        
        response = await self.openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        
        return response.text
    
    async def _stt_whisper(self, audio_base64: str) -> str:
        """Local Whisper STT"""
        # TODO: Implement local Whisper
        logger.warning("Local Whisper not implemented, using mock")
        return await self._stt_mock(audio_base64)
    
    async def _stt_mock(self, audio_base64: str) -> str:
        """Mock STT for testing"""
        return "I'm feeling okay today, just a little tired."
    
    async def _tts_local(self, text: str) -> str:
        """
        Local TTS using pyttsx3 without threading or locks.
        """
        import tempfile
        import os
        import uuid
        
        # Create unique temp file path to avoid conflicts
        temp_dir = tempfile.gettempdir()
        unique_id = uuid.uuid4().hex
        temp_path = os.path.join(temp_dir, f"tts_{unique_id}.wav")
        
        try:
            # Get the pyttsx3 engine
            engine = _get_tts_engine()
            
            # Run the synthesis
            _synthesize_speech(engine, text, temp_path)
            
            # Check if file was actually created
            if not os.path.exists(temp_path):
                logger.error("TTS file was not created")
                return await self._tts_mock(text)
            
            # Read the generated audio file
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Encode to base64
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            return audio_base64
            
        except Exception as e:
            logger.error(f"Local TTS failed: {e}", exc_info=True)
            return await self._tts_mock(text)
            
        finally:
            # Clean up temp file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Could not delete temp file {temp_path}: {e}")
    
    async def _tts_openai(self, text: str) -> str:
        """OpenAI TTS"""
        response = await self.openai_client.audio.speech.create(
            model="tts-1",
            voice=self.tts_voice,
            input=text
        )
        
        # Get audio bytes and encode to base64
        audio_bytes = response.content
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    async def _tts_elevenlabs(self, text: str) -> str:
        """ElevenLabs TTS"""
        # TODO: Implement ElevenLabs
        logger.warning("ElevenLabs TTS not implemented, using local")
        return await self._tts_local(text)
    
    async def _tts_mock(self, text: str) -> str:
        """Mock TTS for testing"""
        # Return base64 of a silent audio or placeholder
        return "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAAB9AAACABAAZGF0YQAAAAA="  # Minimal WAV

speech_service = SpeechService()
