from fastapi import APIRouter, HTTPException
from server.models.schemas import HealthDataEntry
from server.ai.symptom_analyzer import symptom_analyzer
from server.ai.recommendation import recommendation_engine
from server.utils.json_handler import load_json_file, append_to_json_file
from server.config import settings
from typing import Optional, List
from uuid import UUID
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/health", tags=["health"])

@router.post("/log", response_model=HealthDataEntry)
async def log_health_data(entry: HealthDataEntry):
    """
    Log health data entry manually
    
    - Accepts structured health data
    - Validates and stores entry
    - Returns saved entry with ID
    """
    try:
        # Save to health logs
        entry_dict = entry.model_dump(mode='json')
        
        await append_to_json_file(
            f"{settings.data_dir}/health_logs.json",
            entry_dict,
            max_entries=1000
        )
        
        logger.info(f"Logged health data entry {entry.entry_id}")
        
        return entry
    except Exception as e:
        logger.error(f"Error logging health data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log health data: {str(e)}")

@router.get("/logs")
async def get_health_logs(
    limit: int = 50,
    medication_id: Optional[UUID] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """
    Retrieve health logs with optional filtering
    """
    try:
        # Load all logs
        logs = await load_json_file(f"{settings.data_dir}/health_logs.json", default=[])
        
        # Filter by medication_id if provided
        if medication_id:
            logs = [log for log in logs if log.get("medication_id") == str(medication_id)]
        
        # Filter by date range
        if start_date:
            logs = [
                log for log in logs
                if datetime.fromisoformat(log["timestamp"]) >= start_date
            ]
        
        if end_date:
            logs = [
                log for log in logs
                if datetime.fromisoformat(log["timestamp"]) <= end_date
            ]
        
        # Sort by timestamp (newest first) and limit
        logs.sort(key=lambda x: x["timestamp"], reverse=True)
        logs = logs[:limit]
        
        return {
            "total": len(logs),
            "logs": logs
        }
    except Exception as e:
        logger.error(f"Error retrieving health logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {str(e)}")

@router.get("/analyze/trends")
async def analyze_health_trends(
    user_id: str,
    days: int = 7
):
    """
    Analyze health trends over time using AI
    
    - Tracks symptom progression
    - Identifies patterns
    - Generates insights
    """
    try:
        # Get symptom trends
        trends = await symptom_analyzer.track_symptom_trends(user_id, days)
        
        # Load recent logs for analysis
        logs = await load_json_file(f"{settings.data_dir}/health_logs.json", default=[])
        
        # Filter logs for time period
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_logs = [
            log for log in logs
            if datetime.fromisoformat(log["timestamp"]) >= cutoff_date
        ]
        
        # Calculate statistics
        total_entries = len(recent_logs)
        symptoms_reported = sum(
            len(log.get("symptoms", {}))
            for log in recent_logs
        )
        
        # Aggregate symptoms
        symptom_frequency = {}
        for log in recent_logs:
            for symptom, value in log.get("symptoms", {}).items():
                if value:  # If symptom is present
                    symptom_frequency[symptom] = symptom_frequency.get(symptom, 0) + 1
        
        # Sort by frequency
        most_common = sorted(
            symptom_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_entries": total_entries,
            "symptoms_reported": symptoms_reported,
            "most_common_symptoms": [
                {"symptom": s, "frequency": f}
                for s, f in most_common
            ],
            "trends": trends
        }
    except Exception as e:
        logger.error(f"Error analyzing trends: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze trends: {str(e)}")

@router.post("/recommend")
async def get_health_recommendations(
    symptoms: dict,
    user_id: Optional[str] = None
):
    """
    Get personalized health recommendations based on symptoms
    
    - Uses AI to generate recommendations
    - Considers user history if available
    - Prioritizes by urgency
    """
    try:
        # Load user history if user_id provided
        user_history = {}
        if user_id:
            logs = await load_json_file(f"{settings.data_dir}/health_logs.json", default=[])
            # Get last 10 entries for user
            user_logs = [log for log in logs][-10:]
            user_history = {
                "recent_symptoms": [log.get("symptoms", {}) for log in user_logs],
                "entry_count": len(user_logs)
            }
        
        # Generate recommendations
        recommendations = await recommendation_engine.generate_recommendations(
            symptoms,
            user_history
        )
        
        # Prioritize
        recommendations = await recommendation_engine.prioritize_recommendations(
            recommendations
        )
        
        return {
            "symptoms": symptoms,
            "recommendations": [
                {
                    "id": str(r.recommendation_id),
                    "text": r.recommendation_text,
                    "category": r.category,
                    "priority": r.priority,
                    "based_on": r.based_on
                }
                for r in recommendations
            ],
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

@router.get("/summary/daily")
async def get_daily_health_summary(date: Optional[datetime] = None):
    """
    Get daily health summary with AI insights
    """
    try:
        target_date = date or datetime.utcnow()
        start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        # Load logs for the day
        logs = await load_json_file(f"{settings.data_dir}/health_logs.json", default=[])
        
        daily_logs = [
            log for log in logs
            if start_of_day <= datetime.fromisoformat(log["timestamp"]) < end_of_day
        ]
        
        # Aggregate data
        total_symptoms = sum(len(log.get("symptoms", {})) for log in daily_logs)
        
        # Calculate average sentiment
        sentiments = [log.get("sentiment") for log in daily_logs if log.get("sentiment")]
        sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
        avg_sentiment = sum(sentiment_map.get(s, 0) for s in sentiments) / len(sentiments) if sentiments else 0
        
        return {
            "date": target_date.date().isoformat(),
            "total_entries": len(daily_logs),
            "total_symptoms": total_symptoms,
            "average_sentiment": round(avg_sentiment, 2),
            "sentiment_label": "positive" if avg_sentiment > 0.3 else "negative" if avg_sentiment < -0.3 else "neutral",
            "entries": daily_logs
        }
    except Exception as e:
        logger.error(f"Error generating daily summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")
