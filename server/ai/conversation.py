from server.models.schemas import ConversationSession, ConversationMessage
from server.ai.prompts import get_system_prompt
from server.ai.llm_client import llm_client
from server.config import settings
from typing import Dict, List, Optional
from uuid import UUID
from datetime import datetime
import json
import aiofiles
from pathlib import Path
import logging

from server.ai.symptom_analyzer import symptom_analyzer
from server.ai.recommendation import recommendation_engine
from server.utils.json_handler import save_health_data
from server.models.schemas import (
    ConversationSession,
    ConversationMessage,
    SymptomExtractionResult,
    HealthRecommendation,
    HealthDataEntry,
    AIInteraction,
    HealthSymptoms,
    VitalSigns
)

logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self):
        self.active_sessions: Dict[UUID, ConversationSession] = {}
        self.conversation_dir = Path("data/conversations")
        self.conversation_dir.mkdir(parents=True, exist_ok=True)

    async def start_session(
        self,
        medication_id: Optional[UUID] = None
    ) -> ConversationSession:
        """Start a new conversation session"""
        session = ConversationSession(
            medication_id=medication_id
        )

        # Add system prompt
        system_message = ConversationMessage(
            role="system",
            content=get_system_prompt(medication_id)
        )
        session.messages.append(system_message)

        # Store in active sessions
        self.active_sessions[session.session_id] = session
        
        logger.info(f"Started conversation session {session.session_id}")
        
        return session

    async def add_message(
        self,
        session_id: UUID,
        role: str,
        content: str
    ) -> ConversationMessage:
        """Add a message to the conversation"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        message = ConversationMessage(
            role=role,
            content=content
        )
        
        session.messages.append(message)
        
        logger.debug(f"Added {role} message to session {session_id}")
        
        return message

    async def get_context(
        self,
        session_id: UUID,
        max_messages: int = None
    ) -> List[Dict[str, str]]:
        """Get recent conversation context for LLM"""
        max_messages = max_messages or settings.conversation_context_window
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Always include system message + recent messages
        system_msg = next((m for m in session.messages if m.role == "system"), None)
        recent_messages = [m for m in session.messages if m.role != "system"][-max_messages:]
        
        context = []
        if system_msg:
            context.append({"role": system_msg.role, "content": system_msg.content})
        
        for msg in recent_messages:
            context.append({"role": msg.role, "content": msg.content})
        
        return context

    async def generate_response(
        self,
        session_id: UUID,
        user_message: str
    ) -> str:
        """Generate AI response to user message"""
        # Add user message
        await self.add_message(session_id, "user", user_message)

        # Get context
        context = await self.get_context(session_id)

        # Generate response
        try:
            ai_response = await llm_client.generate_response(
                context,
                temperature=settings.llm_temperature,
                # max_tokens=settings.llm_max_tokens
            )
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            ai_response = "I'm sorry, I'm having trouble right now. Could you please repeat that?"

        # Add AI response
        await self.add_message(session_id, "assistant", ai_response)

        return ai_response

    async def get_session(self, session_id: UUID) -> Optional[ConversationSession]:
        """Get session by ID"""
        return self.active_sessions.get(session_id)

    async def end_session(self, session_id: UUID) -> ConversationSession:
        """End conversation and save to storage"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        session.ended_at = datetime.utcnow()
        
        # Save to file
        await self._save_session(session)
        
        # Calculate sentiment score
        session.sentiment_score = await self._calculate_session_sentiment(session)
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        logger.info(f"Ended conversation session {session_id}")
        
        return session

    async def _save_session(self, session: ConversationSession):
        """Save conversation session to file"""
        filename = f"{session.session_id}.json"
        filepath = self.conversation_dir / filename
        
        # Convert to dict
        session_dict = session.model_dump(mode='json')
        
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(session_dict, indent=2, default=str))
        
        logger.debug(f"Saved conversation session to {filepath}")

    async def _calculate_session_sentiment(self, session: ConversationSession) -> float:
        """Calculate overall sentiment score for session"""
        # Simple sentiment based on keywords in user messages
        user_messages = [m.content.lower() for m in session.messages if m.role == "user"]
        
        if not user_messages:
            return 0.0
        
        positive_keywords = ["good", "great", "better", "well", "fine", "happy", "improved"]
        negative_keywords = ["pain", "bad", "worse", "terrible", "awful", "sick", "hurt", "dizzy"]
        
        positive_count = sum(
            1 for msg in user_messages
            for keyword in positive_keywords
            if keyword in msg
        )
        negative_count = sum(
            1 for msg in user_messages
            for keyword in negative_keywords
            if keyword in msg
        )
        
        total_keywords = positive_count + negative_count
        if total_keywords == 0:
            return 0.0
        
        # Score from -1 (very negative) to 1 (very positive)
        score = (positive_count - negative_count) / total_keywords
        return round(score, 2)

    async def load_session(self, session_id: UUID) -> Optional[ConversationSession]:
        """Load a saved conversation session"""
        filename = f"{session_id}.json"
        filepath = self.conversation_dir / filename
        
        if not filepath.exists():
            return None
        
        async with aiofiles.open(filepath, 'r') as f:
            content = await f.read()
            session_dict = json.loads(content)
        
        session = ConversationSession(**session_dict)
        return session

    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[ConversationSession]:
        """Get recent conversation sessions for a user"""
        sessions = []
        
        # Get all session files
        for filepath in sorted(self.conversation_dir.glob("*.json"), reverse=True):
            try:
                async with aiofiles.open(filepath, 'r') as f:
                    content = await f.read()
                    session_dict = json.loads(content)
                    
                if session_dict.get("user_id") == user_id:
                    sessions.append(ConversationSession(**session_dict))
                    
                if len(sessions) >= limit:
                    break
            except Exception as e:
                logger.warning(f"Error loading session from {filepath}: {e}")
        
        return sessions

    async def end_session_with_analysis(
        self,
        session_id: UUID
    ) -> dict:
        """
        End conversation session with full symptom analysis and recommendations
        
        This is the main method that handles all conversation ending logic.
        Used by both AI conversation endpoint and voice endpoint.
        
        Returns:
            Complete analysis dict with symptoms, recommendations, and health entry
        """
        logger.info(f"Ending and analyzing conversation: {session_id}")
        
        # End the session
        session = await self.end_session(session_id)
        
        # Extract conversation text (exclude system messages)
        conversation_text = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in session.messages
            if msg.role != "system"
        ])
        
        # Extract symptoms using AI
        logger.info("Extracting symptoms from conversation...")
        try:
            symptoms = await symptom_analyzer.extract_symptoms(conversation_text)
        except Exception as e:
            logger.error(f"Symptom extraction failed: {e}", exc_info=True)
            # Create safe fallback
            symptoms = self._create_fallback_symptoms()
        
        # Handle empty symptoms
        if not symptoms or not symptoms.symptoms:
            logger.warning("No symptoms extracted, using default values")
            symptoms = self._create_fallback_symptoms(
                sentiment=getattr(symptoms, 'sentiment', 'neutral')
            )
        
        # Generate recommendations
        logger.info("Generating health recommendations...")
        try:
            recommendations = await recommendation_engine.generate_recommendations(
                symptoms.symptoms,
                {}  # TODO: Get user history from storage
            )
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}", exc_info=True)
            recommendations = self._create_fallback_recommendations()
        
        # Build and save health data entry
        health_entry = self._create_health_entry(session, symptoms)
        await save_health_data(health_entry)
        
        logger.info(f"Conversation analysis complete. Urgency: {symptoms.urgency_level}")
        
        return {
            "session_id": session_id,
            "session": session,
            "symptoms": symptoms,
            "recommendations": recommendations,
            "urgency_level": symptoms.urgency_level,
            "health_entry_id": health_entry.entry_id,
            "conversation_duration": (
                (session.ended_at - session.started_at).total_seconds() 
                if session.ended_at else 0
            ),
            "message_count": len(session.messages),
            "conversation_summary": self._generate_summary(session, symptoms)
        }
    
    def _create_fallback_symptoms(self, sentiment: str = "neutral") -> SymptomExtractionResult:
        """Create fallback symptom result when extraction fails"""
        return SymptomExtractionResult(
            symptoms={"general_feeling": "conversation completed"},
            confidence_scores={},
            extracted_entities=[],
            sentiment=sentiment,
            urgency_level="low",
            extracted_count=0
        )
    
    def _create_fallback_recommendations(self) -> list:
        """Create fallback recommendations when generation fails"""
        return [
            HealthRecommendation(
                recommendation_text="Continue taking your medications as prescribed",
                category="general",
                priority="low",
                based_on=[]
            ),
            HealthRecommendation(
                recommendation_text="If you experience any concerning symptoms, contact your healthcare provider",
                category="seek_help",
                priority="medium",
                based_on=[]
            )
        ]
    
    def _create_health_entry(
        self,
        session: ConversationSession,
        symptoms: SymptomExtractionResult
    ) -> HealthDataEntry:
        """Create health data entry from session and symptoms"""
        return HealthDataEntry(
            medication_id=session.medication_id,
            symptoms=HealthSymptoms(**symptoms.symptoms) if symptoms.symptoms else HealthSymptoms(),
            vital_signs=VitalSigns(
                mood=symptoms.symptoms.get("mood") if symptoms.symptoms else None,
                sleep_quality=symptoms.symptoms.get("sleep_quality") if symptoms.symptoms else None,
                appetite=symptoms.symptoms.get("appetite") if symptoms.symptoms else None
            ),
            ai_interaction=AIInteraction(
                questions_asked=[m.content for m in session.messages if m.role == "assistant"],
                responses_given=[m.content for m in session.messages if m.role == "user"],
                conversation_summary=f"Extracted {len(symptoms.symptoms) if symptoms.symptoms else 0} symptoms. Urgency: {symptoms.urgency_level}. Sentiment: {symptoms.sentiment}"
            )
        )
    
    def _generate_summary(
        self,
        session: ConversationSession,
        symptoms: SymptomExtractionResult
    ) -> str:
        """Generate human-readable conversation summary"""
        user_messages = [m for m in session.messages if m.role == "user"]
        
        summary_parts = [
            f"Conversation with {len(user_messages)} user responses",
            f"Sentiment: {symptoms.sentiment}",
            f"Urgency: {symptoms.urgency_level}"
        ]
        
        if symptoms.symptoms:
            key_symptoms = [k for k, v in symptoms.symptoms.items() if v is True][:3]
            if key_symptoms:
                summary_parts.append(f"Key symptoms: {', '.join(key_symptoms)}")
        
        return ". ".join(summary_parts) + "."

conversation_manager = ConversationManager()
