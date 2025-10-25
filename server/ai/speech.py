from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SpeechService:
    def __init__(self):
        if settings.stt_provider == "openai" or settings.tts_provider == "openai":
            self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        # Create voice recordings directory
        self.recordings_dir = Path("data/voice_recordings")
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

    async def speech_to_text(
        self,
        audio_data: str,  # Base64 encoded
        language: str = "en"
    ) -> str:
        """
        Convert audio to text using Whisper API
        """
        if settings.stt_provider == "mock":
            return self._mock_stt()
        
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data)

            # Create file-like object
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"

            # Use OpenAI Whisper
            transcription = await self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="text"
            )

            logger.info(f"Transcribed audio: {transcription[:50]}...")
            return transcription
        except Exception as e:
            logger.error(f"Error in speech-to-text: {e}")
            raise

    async def text_to_speech(
        self,
        text: str,
        voice: str = None  # alloy, echo, fable, onyx, nova, shimmer
    ) -> str:
        """
        Convert text to speech and return base64 encoded audio
        """
        voice = voice or settings.tts_voice
        
        if settings.tts_provider == "mock":
            return self._mock_tts()
        
        try:
            # Use OpenAI TTS
            response = await self.openai_client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                response_format="mp3"
            )

            # Get audio bytes
            audio_bytes = response.content

            # Encode to base64
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

            logger.info(f"Generated TTS for text: {text[:50]}...")
            return audio_base64
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            raise

    async def save_audio(
        self,
        audio_data: str,
        filename: str
    ) -> str:
        """Save audio file to disk"""
        filepath = self.recordings_dir / filename
        
        try:
            audio_bytes = base64.b64decode(audio_data)

            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(audio_bytes)

            logger.info(f"Saved audio to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            raise

    async def load_audio(self, filename: str) -> str:
        """Load audio file and return as base64"""
        filepath = self.recordings_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filename}")
        
        try:
            async with aiofiles.open(filepath, 'rb') as f:
                audio_bytes = await f.read()
            
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            return audio_base64
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise

    def _mock_stt(self) -> str:
        """Mock speech-to-text for testing"""
        return "I'm feeling okay, just a little tired today."

    def _mock_tts(self) -> str:
        """Mock text-to-speech for testing"""
        # Return minimal base64 encoded mock audio
        mock_audio = b"MOCK_AUDIO_DATA"
        return base64.b64encode(mock_audio).decode('utf-8')

speech_service = SpeechService()
