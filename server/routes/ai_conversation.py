from fastapi import APIRouter, HTTPException
from server.models.schemas import ConversationSession, ConversationMessage
from server.ai.conversation import conversation_manager
from uuid import UUID
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ai", tags=["ai"])

@router.post("/conversation/start", response_model=ConversationSession)
async def start_conversation(medication_id: Optional[UUID] = None):
    """
    Start a new AI conversation session for medication check-in
    
    - Creates new conversation session
    - Initializes with system prompt
    - Generates opening greeting
    - Returns session details
    """
    try:
        logger.info(f"Starting conversation for medication: {medication_id}")
        
        session = await conversation_manager.start_session(medication_id)
        
        # Generate opening greeting
        await conversation_manager.generate_response(
            session.session_id,
            "User has opened the medication box and is ready to talk."
        )
        
        logger.info(f"Conversation started: {session.session_id}")
        return session
        
    except Exception as e:
        logger.error(f"Error starting conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start conversation: {str(e)}")

@router.post("/conversation/{session_id}/message", response_model=ConversationMessage)
async def send_message(session_id: UUID, message: str):
    """
    Send a message in an ongoing conversation
    
    - Adds user message to conversation history
    - Generates AI response using LLM
    - Returns AI response message
    """
    try:
        logger.info(f"Sending message to session {session_id}: {message[:50]}...")
        
        ai_response = await conversation_manager.generate_response(session_id, message)
        
        return ConversationMessage(role="assistant", content=ai_response)
        
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation session not found")
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")

@router.post("/conversation/{session_id}/end")
async def end_conversation(session_id: UUID):
    """
    End conversation with full symptom analysis and recommendations
    
    - Ends conversation session
    - Extracts symptoms using AI
    - Generates personalized recommendations
    - Saves health data entry
    - Returns comprehensive summary
    
    This endpoint uses the centralized conversation manager logic
    that is also used by the voice endpoint.
    """
    try:
        # Use centralized conversation ending logic
        result = await conversation_manager.end_session_with_analysis(session_id)
        
        return result
        
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation session not found")
    except Exception as e:
        logger.error(f"Error ending conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to end conversation: {str(e)}")

@router.get("/conversation/{session_id}", response_model=ConversationSession)
async def get_conversation(session_id: UUID):
    """Get conversation details and history"""
    if session_id not in conversation_manager.active_sessions:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversation_manager.active_sessions[session_id]

@router.get("/conversations/active", response_model=List[ConversationSession])
async def list_active_conversations():
    """List all active conversation sessions"""
    return list(conversation_manager.active_sessions.values())

@router.delete("/conversation/{session_id}")
async def delete_conversation(session_id: UUID):
    """Delete/abort a conversation session"""
    try:
        if session_id in conversation_manager.active_sessions:
            del conversation_manager.active_sessions[session_id]
            logger.info(f"Deleted conversation: {session_id}")
            return {"status": "deleted", "session_id": session_id}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
