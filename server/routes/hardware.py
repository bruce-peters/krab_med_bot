"""
Hardware Control API Endpoints
Handles servo motor and LED control through external hardware controller
"""

from fastapi import APIRouter, HTTPException, Request
from datetime import datetime
from typing import Dict, Any
import logging

from server.models.schemas import (
    DispenseRequest,
    DispensingEvent,
    HardwareStatus,
    SuccessResponse,
    ErrorResponse
)

from server.ai.conversation import conversation_manager
from server.utils.json_handler import append_to_json_file
from server.config import settings
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/hardware", tags=["hardware"])


@router.post("/dispense", response_model=DispensingEvent)
async def dispense_medication(request_body: DispenseRequest, api_request: Request):
    """
    Dispense medication from specified compartment
    
    - Activates LED for visual guidance
    - Opens compartment via servo controller
    - Starts AI conversation for health check-in
    - Logs dispensing event
    """
    event_id = uuid4()
    hardware = api_request.app.state.hardware
    
    try:
        logger.info(f"Dispensing medication from compartment {request_body.compartment}")
        
        # Activate LED for compartment
        await hardware.turn_on_led(request_body.compartment)
        logger.debug(f"LED activated for compartment {request_body.compartment}")
        
        # Send command to servo controller (open box)
        servo_response = await hardware.open_servo(request_body.compartment)
        
        if servo_response.get("status") != "success":
            await hardware.show_error_pattern(request_body.compartment)
            raise Exception(f"Servo open failed: {servo_response.get('message')}")
        
        logger.info("Servo opened successfully")
        
        # Start AI conversation automatically
        conversation_session = None
        ai_greeting = None
        
        try:
            conversation_session = await conversation_manager.start_session(
                user_id="current_user",  # TODO: Get from authentication
                medication_id=request_body.medication_id
            )
            
            # Generate greeting
            ai_greeting = await conversation_manager.generate_response(
                conversation_session.session_id,
                "User is taking medication now from the box."
            )
            
            logger.info(f"Started AI conversation {conversation_session.session_id}")
        except Exception as e:
            logger.warning(f"Failed to start AI conversation: {e}")
            # Continue without AI - not critical
        
        # Show success pattern
        await hardware.show_success_pattern(request_body.compartment)
        
        # Create dispensing event
        event = DispensingEvent(
            event_id=event_id,
            timestamp=datetime.utcnow(),
            compartment=request_body.compartment,
            medication_id=request_body.medication_id,
            status="success",
            box_opened=True,
            led_activated=True,
            servo_response=str(servo_response)
        )
        
        # Add AI conversation info to event
        event_dict = event.model_dump()
        event_dict["conversation_session_id"] = str(conversation_session.session_id) if conversation_session else None
        event_dict["ai_greeting"] = ai_greeting
        
        # Save event to logs
        await append_to_json_file(
            f"{settings.data_dir}/dispensing_events.json",
            event_dict,
            max_entries=1000
        )
        
        # Return event with AI info
        return {
            **event_dict,
            "conversation_started": conversation_session is not None
        }
        
    except Exception as e:
        logger.error(f"Error dispensing medication: {e}")
        
        # Show error pattern
        await hardware.show_error_pattern(request_body.compartment)
        await hardware.clear_all_leds()
        
        # Create failed event
        failed_event = DispensingEvent(
            event_id=event_id,
            timestamp=datetime.utcnow(),
            compartment=request_body.compartment,
            medication_id=request_body.medication_id,
            status="failed",
            box_opened=False,
            led_activated=True,
            error_message=str(e)
        )
        
        # Save failed event
        await append_to_json_file(
            f"{settings.data_dir}/dispensing_events.json",
            failed_event.model_dump(),
            max_entries=1000
        )
        
        raise HTTPException(status_code=500, detail=f"Dispensing failed: {str(e)}")


@router.post("/close")
async def close_box(request: Request):
    """
    Close the medication box
    
    - Sends close command to servo
    - Clears all LEDs
    """
    hardware = request.app.state.hardware
    
    try:
        # Clear LEDs first
        await hardware.clear_all_leds()
        
        # Send close command to servo
        servo_response = await hardware.close_servo()
        
        if servo_response.get("status") != "success":
            raise Exception(f"Servo close failed: {servo_response.get('message')}")
        
        logger.info("Box closed successfully")
        
        return {
            "status": "success",
            "message": "Box closed",
            "servo_response": servo_response
        }
    except Exception as e:
        logger.error(f"Error closing box: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to close box: {str(e)}")


@router.get("/status", response_model=HardwareStatus)
async def get_hardware_status(request: Request):
    """
    Get current hardware status
    
    Returns status of:
    - Servo motor (position, operational status)
    - All LEDs (on/off state for each compartment)
    - Last update timestamp
    """
    hardware = request.app.state.hardware
    
    try:
        # Get servo status
        servo_status = await hardware.get_servo_status()
        
        # Get LED status
        led_status_result = await hardware.get_led_status()
        led_status = led_status_result.get("leds", {})
        
        # Build status response
        status = HardwareStatus(
            servo=servo_status or {},
            leds=led_status,
            last_updated=datetime.utcnow()
        )
        
        return status
        
    except Exception as e:
        logger.error(f"✗ Failed to get hardware status: {e}")
        
        # Return error status
        return HardwareStatus(
            servo={},
            leds={},
            last_updated=datetime.utcnow()
        )


@router.post("/test/led/{compartment}")
async def test_led(compartment: int, request: Request):
    """Test LED for specific compartment"""
    hardware = request.app.state.hardware
    
    try:
        if compartment < 1 or compartment > 4:
            raise HTTPException(status_code=400, detail="Compartment must be 1-4")
        
        # Blink LED
        await hardware.blink_led(compartment, duration=2.0)
        
        return {"status": "success", "compartment": compartment}
    except Exception as e:
        logger.error(f"Error testing LED: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test/servo")
async def test_servo(request: Request):
    """Test servo connection and movement"""
    hardware = request.app.state.hardware
    
    try:
        # Test connection
        connected = await hardware.test_connection()
        
        if not connected:
            raise Exception("Hardware controller not responding")
        
        # Test open
        open_response = await hardware.open_servo(1)
        
        # Wait a moment
        import asyncio
        await asyncio.sleep(2)
        
        # Test close
        close_response = await hardware.close_servo()
        
        return {
            "status": "success",
            "connection": "ok",
            "open_test": open_response,
            "close_test": close_response
        }
    except Exception as e:
        logger.error(f"Error testing servo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/connection")
async def check_hardware_connection(request: Request):
    """
    Check connection to hardware controller
    
    Returns:
        Connection status, URL, and timestamp
    """
    hardware = request.app.state.hardware
    
    try:
        connected = await hardware.test_connection()
        
        return {
            "connected": connected,
            "controller_url": hardware.base_url if hasattr(hardware, 'base_url') else "mock",
            "hardware_mode": hardware.__class__.__name__,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/compartments")
async def get_compartments(request: Request):
    """
    Get information about all compartments
    
    Returns:
        Compartment configuration including LED mappings
    """
    hardware = request.app.state.hardware
    
    try:
        compartments = hardware.get_all_compartments()
        
        return {
            "compartments": compartments,
            "total": len(compartments)
        }
    except Exception as e:
        logger.error(f"Failed to get compartments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compartments/{compartment_id}")
async def get_compartment_info(request: Request, compartment_id: int):
    """
    Get information about a specific compartment
    
    Args:
        compartment_id: Compartment number (1-4)
    
    Returns:
        Compartment details
    """
    hardware = request.app.state.hardware
    
    try:
        if not 1 <= compartment_id <= 4:
            raise HTTPException(
                status_code=400,
                detail="Compartment ID must be between 1 and 4"
            )
        
        info = hardware.get_compartment_info(compartment_id)
        
        return {
            "compartment": compartment_id,
            "info": info
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get compartment info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/led/pattern/{compartment_id}")
async def show_led_pattern(
    request: Request,
    compartment_id: int,
    pattern: str = "success"
):
    """
    Show a specific LED pattern
    
    Args:
        compartment_id: Compartment number (1-4)
        pattern: Pattern type ("success", "error", "warning")
    
    Returns:
        Pattern execution result
    """
    hardware = request.app.state.hardware
    
    try:
        if not 1 <= compartment_id <= 4:
            raise HTTPException(
                status_code=400,
                detail="Compartment ID must be between 1 and 4"
            )
        
        if pattern == "success":
            result = await hardware.show_success_pattern(compartment_id)
        elif pattern == "error":
            result = await hardware.show_error_pattern(compartment_id)
        elif pattern == "warning":
            result = await hardware.show_warning_pattern(compartment_id)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid pattern: {pattern}. Use 'success', 'error', or 'warning'"
            )
        
        return {
            "pattern": pattern,
            "compartment": compartment_id,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to show pattern: {e}")
        raise HTTPException(status_code=503, detail=str(e))
