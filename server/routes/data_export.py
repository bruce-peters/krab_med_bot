from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import os
from pathlib import Path
from datetime import datetime
from server.utils.json_handler import read_json, write_json
from server.config import settings
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/data", tags=["data-export"])

def get_data_path(filename: str) -> Path:
    """Get full path to data file"""
    return Path(settings.data_dir) / filename

def _to_unix_ts(value: Any) -> int:
    """Safely convert value to Unix timestamp"""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(dt.timestamp())
        except ValueError:
            pass  # Invalid format, fallback to 0
    return 0  # Default to 0 if conversion fails

def aggregate_patient_data() -> Dict[str, Any]:
    """Aggregate all JSON data into reference patient data format"""
    
    # Read medication schedule
    medication_schedule = read_json(get_data_path("medication_schedule.json"), default={})
    
    # Read health logs
    health_logs = read_json(get_data_path("health_logs.json"), default={"entries": []})
    
    # Read user interactions
    user_interactions = read_json(get_data_path("user_interactions.json"), default={"interactions": []})
    
    # Extract patient info and pills from medication schedule
    patient_info = {
        "name": medication_schedule.get("patient_name", "Unknown Patient"),
        "age": medication_schedule.get("patient_age", 0),
        "pills": []
    }
    
    # Convert medications to pills format
    for med in medication_schedule.get("medications", []):
        # Get frequency - it should already be in the correct format
        frequency = med.get("frequency", {})
        
        # Ensure all days are present
        default_frequency = {
            "Monday": False,
            "Tuesday": False,
            "Wednesday": False,
            "Thursday": False,
            "Friday": False,
            "Saturday": False,
            "Sunday": False
        }
        default_frequency.update(frequency)
        
        pill = {
            "name": med.get("name", ""),
            "frequency": default_frequency,
            "box_index": med.get("box_index", med.get("compartment", 0))
        }
        patient_info["pills"].append(pill)
    
    # Convert health logs and interactions to transcriptions
    transcriptions = []
    
    # Add health log entries (supports list or dict with 'entries')
    entries = health_logs if isinstance(health_logs, list) else health_logs.get("entries", [])
    for entry in entries:
        timestamp = _to_unix_ts(entry.get("timestamp", 0))
        if timestamp == 0:
            continue  # Skip invalid timestamps
        # Build transcription from symptoms, notes, and AI responses
        parts: List[str] = []
        symptoms = entry.get("symptoms")
        if isinstance(symptoms, dict):
            if symptoms.get("custom_notes"):
                parts.append(symptoms["custom_notes"])
            symptom_list = []
            if symptoms.get("pain_level") is not None:
                symptom_list.append(f"pain level {symptoms.get('pain_level')}")
            for k, label in [
                ("nausea", "nausea"),
                ("dizziness", "dizziness"),
                ("fatigue", "fatigue"),
                ("headache", "headache"),
                ("shortness_of_breath", "shortness of breath")
            ]:
                if symptoms.get(k):
                    symptom_list.append(label)
            if symptom_list:
                parts.append(", ".join(symptom_list))
        elif isinstance(symptoms, str):
            parts.append(symptoms)
        if entry.get("notes"):
            parts.append(entry["notes"])
        
        # Include AI interaction responses
        ai_interaction = entry.get("ai_interaction")
        if ai_interaction and isinstance(ai_interaction, dict):
            responses = ai_interaction.get("responses_given", [])
            if responses:
                parts.append(" ".join(responses))
        
        text = ". ".join(p for p in parts if p)
        if text:
            transcriptions.append({"timestamp": timestamp, "transcription": text})
    
    # Add user interaction transcripts (supports list or dict with 'interactions')
    interactions = user_interactions if isinstance(user_interactions, list) else user_interactions.get("interactions", [])
    for interaction in interactions:
        timestamp = _to_unix_ts(interaction.get("timestamp", 0))
        if timestamp == 0:
            continue  # Skip invalid timestamps
        if interaction.get("type") == "conversation" or interaction.get("messages"):
            messages = interaction.get("messages", [])
            if messages:
                user_messages = [
                    msg.get("content", "")
                    for msg in messages
                    if msg.get("role") == "user" and msg.get("content")
                ]
                if user_messages:
                    transcriptions.append({
                        "timestamp": timestamp,
                        "transcription": " ".join(user_messages)
                    })

    # Sort transcriptions by timestamp
    transcriptions.sort(key=lambda x: x.get("timestamp", 0))

    # Return aggregated patient data
    return {
        "patient_info": patient_info,
        "transcriptions": transcriptions
    }

@router.get("/export/patient-data", response_model=Dict[str, Any])
async def export_patient_data():
    """
    Export all project data in reference patient data format.
    
    Returns:
        Aggregated data from:
        - medication_schedule.json
        - health_logs.json
        - user_interactions.json
    """
    try:
        aggregated_data = aggregate_patient_data()
        return aggregated_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to aggregate patient data: {str(e)}")

@router.post("/export/patient-data/save", response_model=Dict[str, Any])
async def save_patient_data_export(filename: str = "exported_patient_data.json"):
    """
    Export and save patient data to a file.
    
    Args:
        filename: Name of the file to save (default: exported_patient_data.json)
    
    Returns:
        Success message with file path and record counts.
    """
    try:
        aggregated_data = aggregate_patient_data()
        file_path = get_data_path(filename)
        write_json(file_path, aggregated_data)
        return {
            "status": "success",
            "message": f"Patient data exported successfully",
            "file_path": str(file_path),
            "record_count": {
                "pills": len(aggregated_data["patient_info"]["pills"]),
                "transcriptions": len(aggregated_data["transcriptions"])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save patient data: {str(e)}")
