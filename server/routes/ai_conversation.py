from fastapi import APIRouter, HTTPException
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
from server.ai.conversation import conversation_manager
from server.ai.symptom_analyzer import symptom_analyzer
from server.ai.recommendation import recommendation_engine
from server.utils.json_handler import save_health_data
from uuid import UUID
from typing import Optional, List
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
    
    - Creates new conversation session
    - Initializes with system prompt
    - Generates opening greeting
    - Returns session details
    """
    try:
        logger.info(f"Starting conversation for user: {user_id}, medication: {medication_id}")
        
        session = await conversation_manager.start_session(user_id, medication_id)
        
        # Generate opening greeting (internal trigger)
        greeting = await conversation_manager.generate_response(
            session.session_id,
            "User has opened the medication box and is ready to talk."
        )
        
        logger.info(f"Conversation started: {session.session_id}")
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
    
    - Adds user message to conversation history
    - Generates AI response using LLM
    - Returns AI response message
    """
    try:
        logger.info(f"Sending message to session {session_id}: {message[:50]}...")
        
        ai_response = await conversation_manager.generate_response(
            session_id,
            message
        )
        
        return ConversationMessage(
            role="assistant",
            content=ai_response
        )
        
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation session not found")
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")

@router.post("/conversation/{session_id}/end")
async def end_conversation(session_id: UUID):
    """
    End conversation and extract symptoms/generate recommendations
    
    - Ends conversation session
    - Extracts all conversation text
    - Analyzes for symptoms using AI
    - Generates personalized recommendations
    - Saves health data entry
    - Returns comprehensive summary
    """
    try:
        logger.info(f"Ending conversation: {session_id}")
        
        session = await conversation_manager.end_session(session_id)
        
        # Extract conversation text (exclude system messages)
        conversation_text = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in session.messages
            if msg.role != "system"
        ])
        
        # Extract symptoms using AI
        logger.info("Extracting symptoms from conversation...")
        symptoms = await symptom_analyzer.extract_symptoms(conversation_text)
        
        # Generate recommendations
        logger.info("Generating health recommendations...")
        recommendations = await recommendation_engine.generate_recommendations(
            symptoms.symptoms,
            {}  # TODO: Get user history from storage
        )
        
        # Build health data entry
        health_entry = HealthDataEntry(
            medication_id=session.medication_id,
            symptoms=HealthSymptoms(**symptoms.symptoms),
            vital_signs=VitalSigns(
                mood=symptoms.symptoms.get("mood", "okay"),
                sleep_quality=symptoms.symptoms.get("sleep_quality", "fair")
            ),
            ai_interaction=AIInteraction(
                questions_asked=[m.content for m in session.messages if m.role == "assistant"],
                responses_given=[m.content for m in session.messages if m.role == "user"],
                conversation_summary=f"Extracted {len(symptoms.symptoms)} symptoms. Urgency: {symptoms.urgency_level}"
            )
        )
        
        # Save to health logs
        await save_health_data(health_entry)
        
        logger.info(f"Conversation ended successfully. Urgency: {symptoms.urgency_level}")
        
        return {
            "session_id": session_id,
            "symptoms": symptoms,
            "recommendations": recommendations,
            "urgency_level": symptoms.urgency_level,
            "health_entry_id": health_entry.entry_id,
            "conversation_duration": (session.ended_at - session.started_at).total_seconds() if session.ended_at else 0
        }
        
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation session not found")
    except Exception as e:
        logger.error(f"Error ending conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to end conversation: {str(e)}")

@router.get("/conversation/{session_id}", response_model=ConversationSession)
async def get_conversation(session_id: UUID):
    """
    Get conversation details and history
    
    - Returns full conversation session
    - Includes all messages
    - Shows current status
    """
    if session_id not in conversation_manager.active_sessions:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversation_manager.active_sessions[session_id]

@router.get("/conversations/active", response_model=List[ConversationSession])
async def list_active_conversations():
    """
    List all active conversation sessions
    
    - Returns all ongoing conversations
    - Useful for monitoring/debugging
    """
    return list(conversation_manager.active_sessions.values())

@router.delete("/conversation/{session_id}")
async def delete_conversation(session_id: UUID):
    """
    Delete/abort a conversation session
    
    - Removes session without saving
    - Use for error recovery
    """
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
