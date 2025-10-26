import json
import aiofiles
from pathlib import Path
from typing import Any, List, Dict, Optional
from datetime import datetime
from server.config import settings
import logging

from server.models.schemas import HealthDataEntry, DispensingEvent

logger = logging.getLogger(__name__)

async def load_json_file(filepath: str, default: Any = None) -> Any:
    """
    Load data from a JSON file asynchronously.
    
    Args:
        filepath: Path to the JSON file
        default: Default value to return if file doesn't exist
        
    Returns:
        Loaded JSON data or default value
    """
    path = Path(filepath)
    
    if not path.exists():
        logger.warning(f"JSON file not found: {filepath}, using default value")
        return default if default is not None else []
    
    try:
        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            content = await f.read()
            return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}")
        return default if default is not None else []
    except Exception as e:
        logger.error(f"Error loading JSON file {filepath}: {e}")
        return default if default is not None else []

async def save_json_file(filepath: str, data: Any, indent: int = 2) -> bool:
    """
    Save data to a JSON file asynchronously.
    
    Args:
        filepath: Path to the JSON file
        data: Data to save
        indent: JSON indentation level
        
    Returns:
        True if successful, False otherwise
    """
    path = Path(filepath)
    
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            json_str = json.dumps(data, indent=indent, default=str, ensure_ascii=False)
            await f.write(json_str)
        logger.debug(f"Successfully saved data to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {filepath}: {e}")
        return False

async def append_to_json_file(filepath: str, entry: Dict, max_entries: Optional[int] = None) -> bool:
    """
    Append an entry to a JSON file (assuming file contains a list).
    
    Args:
        filepath: Path to the JSON file
        entry: Entry to append
        max_entries: Maximum number of entries to keep (oldest removed first)
        
    Returns:
        True if successful, False otherwise
    """
    # Load existing data
    data = await load_json_file(filepath, default=[])
    
    if not isinstance(data, list):
        logger.error(f"Cannot append to {filepath}: file does not contain a list")
        return False
    
    # Append new entry
    data.append(entry)
    
    # Trim if max_entries specified
    if max_entries and len(data) > max_entries:
        data = data[-max_entries:]
    
    # Save back
    return await save_json_file(filepath, data)

async def update_json_entry(
    filepath: str,
    entry_id: str,
    updates: Dict,
    id_field: str = "id"
) -> bool:
    """
    Update a specific entry in a JSON file (assuming file contains a list).
    
    Args:
        filepath: Path to the JSON file
        entry_id: ID of the entry to update
        updates: Dictionary of fields to update
        id_field: Name of the ID field
        
    Returns:
        True if successful, False otherwise
    """
    # Load existing data
    data = await load_json_file(filepath, default=[])
    
    if not isinstance(data, list):
        logger.error(f"Cannot update in {filepath}: file does not contain a list")
        return False
    
    # Find and update entry
    updated = False
    for entry in data:
        if isinstance(entry, dict) and str(entry.get(id_field)) == str(entry_id):
            entry.update(updates)
            entry["updated_at"] = datetime.utcnow().isoformat()
            updated = True
            break
    
    if not updated:
        logger.warning(f"Entry with {id_field}={entry_id} not found in {filepath}")
        return False
    
    # Save back
    return await save_json_file(filepath, data)

async def initialize_data_file(filepath: str, default_data: Any) -> bool:
    """
    Initialize a data file if it doesn't exist.
    
    Args:
        filepath: Path to the JSON file
        default_data: Default data structure to create
        
    Returns:
        True if file was created, False if it already existed
    """
    path = Path(filepath)
    
    if path.exists():
        logger.debug(f"Data file already exists: {filepath}")
        return False
    
    # Create parent directories
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save default data
    success = await save_json_file(filepath, default_data)
    
    if success:
        logger.info(f"Initialized data file: {filepath}")
    
    return success

def load_json_file_sync(filepath: str, default: Any = None) -> Any:
    """
    Synchronous version of load_json_file for non-async contexts.
    """
    path = Path(filepath)
    
    if not path.exists():
        logger.warning(f"JSON file not found: {filepath}, using default value")
        return default if default is not None else []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}")
        return default if default is not None else []
    except Exception as e:
        logger.error(f"Error loading JSON file {filepath}: {e}")
        return default if default is not None else []

def save_json_file_sync(filepath: str, data: Any, indent: int = 2) -> bool:
    """
    Synchronous version of save_json_file for non-async contexts.
    """
    path = Path(filepath)
    
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, default=str, ensure_ascii=False)
        logger.debug(f"Successfully saved data to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {filepath}: {e}")
        return False

async def save_health_data(entry: HealthDataEntry) -> None:
    """Save health data entry to JSON file"""
    filepath = Path(settings.data_dir) / "health_logs.json"
    
    async with aiofiles.open(filepath, mode='r') as f:
        content = await f.read()
        data = json.loads(content) if content else []
    
    # Convert to dict and add to list
    data.append(entry.dict())
    
    async with aiofiles.open(filepath, mode='w') as f:
        await f.write(json.dumps(data, indent=2, default=str))

async def load_health_data() -> List[HealthDataEntry]:
    """Load all health data entries"""
    filepath = Path(settings.data_dir) / "health_logs.json"
    
    if not filepath.exists():
        return []
    
    async with aiofiles.open(filepath, mode='r') as f:
        content = await f.read()
        data = json.loads(content) if content else []
    
    return [HealthDataEntry(**entry) for entry in data]

async def save_dispensing_event(event: DispensingEvent) -> None:
    """Save dispensing event to JSON file"""
    filepath = Path(settings.data_dir) / "dispensing_events.json"
    
    async with aiofiles.open(filepath, mode='r') as f:
        content = await f.read()
        data = json.loads(content) if content else []
    
    data.append(event.dict())
    
    async with aiofiles.open(filepath, mode='w') as f:
        await f.write(json.dumps(data, indent=2, default=str))

def read_json(file_path: Path, default: Optional[Any] = None) -> Any:
    """
    Read a JSON file and return its contents.
    If the file does not exist, return the default value.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON from {file_path}: {e}")

def write_json(file_path: Path, data: Any) -> None:
    """
    Write data to a JSON file.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        raise IOError(f"Failed to write JSON to {file_path}: {e}")
