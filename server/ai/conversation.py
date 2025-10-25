from server.models.schemas import ConversationSession, ConversationMessage
from server.ai.prompts import get_system_prompt
from server.ai.llm_client import llm_client
from typing import Dict, List, Optional
from uuid import UUID
from datetime import datetime
import json
import aiofiles
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self):
        self.active_sessions: Dict[UUID, ConversationSession] = {}
        self.conversation_dir = Path("data/conversations")
        self.conversation_dir.mkdir(parents=True, exist_ok=True)

    async def start_session(
        self,
        user_id: str,
        medication_id: Optional[UUID] = None
    ) -> ConversationSession:
        """Start a new conversation session"""
        session = ConversationSession(
            user_id=user_id,
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
        
        logger.info(f"Started conversation session {session.session_id} for user {user_id}")
        
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
                max_tokens=settings.llm_max_tokens
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

conversation_manager = ConversationManager()
