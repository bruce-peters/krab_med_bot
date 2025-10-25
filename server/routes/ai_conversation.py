from fastapi import APIRouter, HTTPException
from server.models.schemas import (
    ConversationSession,
    ConversationMessage,
    HealthDataEntry,
    AIInteraction,
    HealthSymptoms,
    VitalSigns
)
from server.ai.conversation import conversation_manager
from server.ai.symptom_analyzer import symptom_analyzer
from server.ai.recommendation import recommendation_engine
from server.utils.json_handler import append_to_json_file
from server.config import settings
from typing import Optional, List
from uuid import UUID
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ai", tags=["ai"])

@router.post("/conversation/start", response_model=ConversationSession)
async def start_conversation(
    user_id: str,
    medication_id: Optional[UUID] = None
):
    """
    Start a new AI conversation session for medication check-in
    
    - Initializes conversation context
    - Generates warm greeting
    - Returns session details for frontend to track
    """
    try:
        session = await conversation_manager.start_session(user_id, medication_id)
        
        # Generate opening greeting
        greeting = await conversation_manager.generate_response(
            session.session_id,
            "User has opened the medication box and is ready to talk."
        )
        
        logger.info(f"Started conversation {session.session_id} with greeting: {greeting[:50]}...")
        
        return session
    except Exception as e:
        logger.error(f"Error starting conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start conversation: {str(e)}")

@router.post("/conversation/{session_id}/message", response_model=ConversationMessage)
async def send_message(
    session_id: UUID,
    message: str
):
    """
    Send a message in an ongoing conversation
    
    - Adds user message to conversation
    - Generates AI response
    - Maintains conversation context
    """
    try:
        ai_response = await conversation_manager.generate_response(
            session_id,
            message
        )
        
        return ConversationMessage(
            role="assistant",
            content=ai_response
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

@router.get("/conversation/{session_id}", response_model=ConversationSession)
async def get_conversation(session_id: UUID):
    """
    Get conversation details and full message history
    """
    session = await conversation_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return session

@router.post("/conversation/{session_id}/end")
async def end_conversation(session_id: UUID):
    """
    End conversation and extract symptoms/generate recommendations
    
    - Analyzes full conversation for symptoms
    - Generates personalized health recommendations
    - Saves health data entry
    - Returns comprehensive summary
    """
    try:
        session = await conversation_manager.end_session(session_id)
        
        # Extract conversation text (exclude system messages)
        conversation_text = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in session.messages
            if msg.role != "system"
        ])
        
        # Extract symptoms using AI
        symptom_result = await symptom_analyzer.extract_symptoms(conversation_text)
        
        # Generate recommendations based on symptoms
        recommendations = []
        if symptom_result.symptoms:
            recommendation_list = await recommendation_engine.generate_recommendations(
                symptom_result.symptoms,
                {}  # TODO: Load user history from database
            )
            recommendations = [
                {
                    "text": r.recommendation_text,
                    "category": r.category,
                    "priority": r.priority
                }
                for r in recommendation_list
            ]
        
        # Create health data entry
        from uuid import uuid4
        health_entry = {
            "entry_id": str(uuid4()),
            "timestamp": session.ended_at.isoformat() if session.ended_at else None,
            "medication_id": str(session.medication_id) if session.medication_id else None,
            "symptoms": symptom_result.symptoms,
            "vital_signs": {},  # Could be extracted from symptoms
            "ai_interaction": {
                "questions_asked": [m.content for m in session.messages if m.role == "assistant"],
                "responses_given": [m.content for m in session.messages if m.role == "user"],
                "conversation_summary": f"Extracted {len(symptom_result.symptoms)} symptoms. Urgency: {symptom_result.urgency_level}"
            },
            "sentiment": symptom_result.sentiment,
            "urgency_level": symptom_result.urgency_level,
            "session_id": str(session.session_id)
        }
        
        # Save to health logs
        await append_to_json_file(
            f"{settings.data_dir}/health_logs.json",
            health_entry,
            max_entries=1000
        )
        
        logger.info(f"Ended conversation {session_id}. Extracted {len(symptom_result.symptoms)} symptoms.")
        
        return {
            "session_id": str(session_id),
            "symptoms": symptom_result.symptoms,
            "confidence_scores": symptom_result.confidence_scores,
            "recommendations": recommendations,
            "urgency_level": symptom_result.urgency_level,
            "sentiment": symptom_result.sentiment,
            "sentiment_score": session.sentiment_score,
            "message_count": len(session.messages),
            "health_entry_saved": True
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error ending conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to end conversation: {str(e)}")

@router.get("/conversation/user/{user_id}/history")
async def get_user_conversation_history(
    user_id: str,
    limit: int = 10
):
    """
    Get recent conversation history for a user
    """
    try:
        sessions = await conversation_manager.get_user_sessions(user_id, limit)
        
        return {
            "user_id": user_id,
            "total_sessions": len(sessions),
            "sessions": [
                {
                    "session_id": str(s.session_id),
                    "started_at": s.started_at.isoformat(),
                    "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                    "message_count": len(s.messages),
                    "sentiment_score": s.sentiment_score
                }
                for s in sessions
            ]
        }
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

@router.delete("/conversation/{session_id}")
async def delete_conversation(session_id: UUID):
    """
    Delete a conversation session (soft delete - archive)
    """
    try:
        session = await conversation_manager.get_session(session_id)
        
        if not session:
            # Try loading from file
            session = await conversation_manager.load_session(session_id)
            
        if not session:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # In production, move to archive instead of deleting
        logger.info(f"Archived conversation {session_id}")
        
        return {"message": "Conversation archived", "session_id": str(session_id)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")
