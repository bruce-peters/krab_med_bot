"""
Medication Management API Endpoints
Handles medication schedules, tracking, and history
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from datetime import date, datetime
from uuid import UUID
import logging

from server.models.schemas import (
    MedicationEntry,
    MedicationSchedule,
    MarkTakenRequest,
    SuccessResponse
)
from server.utils.json_handler import (
    read_json,
    write_json,
    append_to_json_array,
    update_json_entry
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/medications", tags=["medications"])

# Data file path
MEDICATION_FILE = "data/medication_schedule.json"


@router.get("/", response_model=List[MedicationEntry])
async def get_medications(filter_date: Optional[date] = Query(None, alias="date")):
    """
    Get medication schedule
    
    Args:
        filter_date: Optional date to filter medications (YYYY-MM-DD)
    
    Returns:
        List of medication entries
    """
    try:
        data = await read_json(MEDICATION_FILE)
        medications = data.get("medications", [])
        
        # Convert to MedicationEntry objects
        med_entries = []
        for med in medications:
            # Convert string timestamps to datetime if needed
            if isinstance(med.get("scheduled_time"), str):
                med["scheduled_time"] = datetime.fromisoformat(med["scheduled_time"].replace("Z", "+00:00"))
            if med.get("taken_timestamp") and isinstance(med["taken_timestamp"], str):
                med["taken_timestamp"] = datetime.fromisoformat(med["taken_timestamp"].replace("Z", "+00:00"))
            
            entry = MedicationEntry(**med)
            
            # Filter by date if provided
            if filter_date:
                if entry.scheduled_time.date() == filter_date:
                    med_entries.append(entry)
            else:
                med_entries.append(entry)
        
        logger.info(f"Retrieved {len(med_entries)} medications")
        return med_entries
        
    except FileNotFoundError:
        logger.warning(f"Medication file not found: {MEDICATION_FILE}")
        return []
    except Exception as e:
        logger.error(f"Failed to get medications: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/next", response_model=Optional[MedicationEntry])
async def get_next_medication():
    """
    Get the next scheduled medication
    
    Returns:
        Next medication entry or None if no upcoming medications
    """
    try:
        data = await read_json(MEDICATION_FILE)
        medications = data.get("medications", [])
        
        now = datetime.utcnow()
        next_med = None
        min_diff = None
        
        for med in medications:
            # Skip if already taken
            if med.get("taken", False):
                continue
            
            # Parse scheduled time
            sched_time = med.get("scheduled_time")
            if isinstance(sched_time, str):
                sched_time = datetime.fromisoformat(sched_time.replace("Z", "+00:00"))
            
            # Calculate time difference
            time_diff = (sched_time - now).total_seconds()
            
            # Only consider future medications
            if time_diff > 0:
                if min_diff is None or time_diff < min_diff:
                    min_diff = time_diff
                    next_med = med
        
        if next_med:
            if isinstance(next_med.get("scheduled_time"), str):
                next_med["scheduled_time"] = datetime.fromisoformat(next_med["scheduled_time"].replace("Z", "+00:00"))
            
            entry = MedicationEntry(**next_med)
            logger.info(f"Next medication: {entry.name} at {entry.scheduled_time}")
            return entry
        else:
            logger.info("No upcoming medications")
            return None
            
    except FileNotFoundError:
        logger.warning(f"Medication file not found: {MEDICATION_FILE}")
        return None
    except Exception as e:
        logger.error(f"Failed to get next medication: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mark-taken", response_model=MedicationEntry)
async def mark_medication_taken(mark_req: MarkTakenRequest):
    """
    Mark a medication as taken
    
    Args:
        mark_req: Contains medication ID and optional timestamp
    
    Returns:
        Updated medication entry
    """
    try:
        data = await read_json(MEDICATION_FILE)
        medications = data.get("medications", [])
        
        # Find the medication
        med_found = False
        for med in medications:
            if str(med.get("medication_id")) == str(mark_req.medication_id):
                med["taken"] = True
                med["taken_timestamp"] = (
                    mark_req.timestamp.isoformat() if mark_req.timestamp
                    else datetime.utcnow().isoformat()
                )
                med_found = True
                
                # Save updated data
                await write_json(MEDICATION_FILE, data)
                
                # Return updated entry
                if isinstance(med.get("scheduled_time"), str):
                    med["scheduled_time"] = datetime.fromisoformat(med["scheduled_time"].replace("Z", "+00:00"))
                if isinstance(med.get("taken_timestamp"), str):
                    med["taken_timestamp"] = datetime.fromisoformat(med["taken_timestamp"].replace("Z", "+00:00"))
                
                entry = MedicationEntry(**med)
                logger.info(f"✓ Marked medication {entry.name} as taken")
                return entry
        
        if not med_found:
            raise HTTPException(
                status_code=404,
                detail=f"Medication not found: {mark_req.medication_id}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to mark medication as taken: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=List[MedicationEntry])
async def get_medication_history(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """
    Get medication history with optional date filtering
    
    Args:
        start_date: Start date for filtering (YYYY-MM-DD)
        end_date: End date for filtering (YYYY-MM-DD)
        limit: Maximum number of results (1-1000)
    
    Returns:
        List of medication entries matching filters
    """
    try:
        data = await read_json(MEDICATION_FILE)
        medications = data.get("medications", [])
        
        # Filter medications
        filtered = []
        for med in medications:
            # Parse scheduled time
            sched_time = med.get("scheduled_time")
            if isinstance(sched_time, str):
                sched_time = datetime.fromisoformat(sched_time.replace("Z", "+00:00"))
            
            # Apply date filters
            if start_date and sched_time.date() < start_date:
                continue
            if end_date and sched_time.date() > end_date:
                continue
            
            # Convert timestamps
            if isinstance(med.get("scheduled_time"), str):
                med["scheduled_time"] = datetime.fromisoformat(med["scheduled_time"].replace("Z", "+00:00"))
            if med.get("taken_timestamp") and isinstance(med["taken_timestamp"], str):
                med["taken_timestamp"] = datetime.fromisoformat(med["taken_timestamp"].replace("Z", "+00:00"))
            
            filtered.append(MedicationEntry(**med))
        
        # Sort by scheduled time (most recent first)
        filtered.sort(key=lambda x: x.scheduled_time, reverse=True)
        
        # Apply limit
        filtered = filtered[:limit]
        
        logger.info(f"Retrieved {len(filtered)} medication history entries")
        return filtered
        
    except FileNotFoundError:
        logger.warning(f"Medication file not found: {MEDICATION_FILE}")
        return []
    except Exception as e:
        logger.error(f"Failed to get medication history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=MedicationEntry)
async def create_medication(medication: MedicationEntry):
    """
    Add a new medication to the schedule
    
    Args:
        medication: Medication entry to add
    
    Returns:
        Created medication entry
    """
    try:
        data = await read_json(MEDICATION_FILE)
        
        # Add to medications list
        med_dict = medication.model_dump(mode="json")
        data["medications"].append(med_dict)
        
        # Save
        await write_json(MEDICATION_FILE, data)
        
        logger.info(f"✓ Created medication: {medication.name}")
        return medication
        
    except Exception as e:
        logger.error(f"Failed to create medication: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{medication_id}", response_model=SuccessResponse)
async def delete_medication(medication_id: UUID):
    """
    Delete a medication from the schedule
    
    Args:
        medication_id: ID of medication to delete
    
    Returns:
        Success response
    """
    try:
        data = await read_json(MEDICATION_FILE)
        medications = data.get("medications", [])
        
        # Find and remove medication
        original_count = len(medications)
        data["medications"] = [
            med for med in medications
            if str(med.get("medication_id")) != str(medication_id)
        ]
        
        if len(data["medications"]) == original_count:
            raise HTTPException(
                status_code=404,
                detail=f"Medication not found: {medication_id}"
            )
        
        # Save
        await write_json(MEDICATION_FILE, data)
        
        logger.info(f"✓ Deleted medication: {medication_id}")
        return SuccessResponse(
            success=True,
            message=f"Medication {medication_id} deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete medication: {e}")
        raise HTTPException(status_code=500, detail=str(e))
