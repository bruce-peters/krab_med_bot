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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/hardware", tags=["hardware"])


@router.post("/dispense", response_model=DispensingEvent)
async def dispense_medication(request: Request, dispense_req: DispenseRequest):
    """
    Dispense medication from specified compartment
    
    Steps:
    1. Turn on LED for the compartment
    2. Open servo motor
    3. Log the dispensing event
    4. Return event details
    
    Args:
        dispense_req: Contains compartment number and medication ID
    
    Returns:
        DispensingEvent with status and details
    """
    hardware = request.app.state.hardware
    
    try:
        logger.info(f"Dispensing medication from compartment {dispense_req.compartment}")
        
        # Step 1: Turn on LED
        led_result = await hardware.turn_on_led(dispense_req.compartment)
        led_activated = led_result.get("status") == "success"
        
        # Step 2: Open servo
        servo_result = await hardware.open_servo(dispense_req.compartment)
        box_opened = servo_result.get("status") == "success"
        
        # Create dispensing event
        event = DispensingEvent(
            compartment=dispense_req.compartment,
            medication_id=dispense_req.medication_id,
            status="success" if (led_activated and box_opened) else "failed",
            box_opened=box_opened,
            led_activated=led_activated,
            servo_response=str(servo_result),
            error_message=None
        )
        
        logger.info(f"✓ Successfully dispensed from compartment {dispense_req.compartment}")
        return event
        
    except Exception as e:
        logger.error(f"✗ Failed to dispense medication: {e}")
        
        # Create failed event
        event = DispensingEvent(
            compartment=dispense_req.compartment,
            medication_id=dispense_req.medication_id,
            status="failed",
            box_opened=False,
            led_activated=False,
            servo_response=None,
            error_message=str(e)
        )
        
        raise HTTPException(
            status_code=503,
            detail=f"Hardware error: {str(e)}"
        )


@router.post("/close", response_model=SuccessResponse)
async def close_box(request: Request):
    """
    Close the medication box
    
    Steps:
    1. Send close command to servo
    2. Turn off all LEDs
    3. Log the closure
    
    Returns:
        Success response with timestamp
    """
    hardware = request.app.state.hardware
    
    try:
        logger.info("Closing medication box")
        
        # Close servo
        servo_result = await hardware.close_servo()
        
        # Turn off all LEDs
        led_result = await hardware.clear_all_leds()
        
        return SuccessResponse(
            success=True,
            message="Medication box closed successfully",
            data={
                "servo_confirmed": servo_result.get("status") == "success",
                "leds_cleared": led_result.get("status") == "success",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"✗ Failed to close box: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Hardware error: {str(e)}"
        )


@router.get("/status", response_model=HardwareStatus)
async def get_hardware_status(request: Request):
    """
    Get current hardware status
    
    Returns status of:
    - Servo motor (position, operational status)
    - All LEDs (on/off state for each compartment)
    - Last update timestamp
    
    Returns:
        HardwareStatus with servo and LED information
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
            servo={
                "position": servo_status.get("position", "unknown"),
                "operational": servo_status.get("operational", False),
                "controller_reachable": True,
                "current_compartment": servo_status.get("current_compartment"),
                "last_response": datetime.utcnow().isoformat()
            },
            leds=led_status,
            last_updated=datetime.utcnow()
        )
        
        return status
        
    except Exception as e:
        logger.error(f"✗ Failed to get hardware status: {e}")
        
        # Return error status
        return HardwareStatus(
            servo={
                "position": "unknown",
                "operational": False,
                "controller_reachable": False,
                "last_response": None
            },
            leds={},
            last_updated=datetime.utcnow()
        )


@router.post("/test")
async def test_hardware(request: Request, component: str, action: str):
    """
    Test individual hardware components
    
    Args:
        component: "servo" or "led"
        action: Component-specific action
            - servo: "open", "close", "status"
            - led: "on:<id>", "off:<id>", "blink:<id>", "clear", "status"
    
    Returns:
        Test results with component response
    """
    hardware = request.app.state.hardware
    
    try:
        result = {}
        
        if component == "servo":
            if action == "open":
                result = await hardware.open_servo(1)
            elif action == "close":
                result = await hardware.close_servo()
            elif action == "status":
                result = await hardware.get_servo_status()
            else:
                raise HTTPException(status_code=400, detail=f"Invalid servo action: {action}")
                
        elif component == "led":
            if action.startswith("on:"):
                led_id = int(action.split(":")[1])
                result = await hardware.turn_on_led(led_id)
            elif action.startswith("off:"):
                led_id = int(action.split(":")[1])
                result = await hardware.turn_off_led(led_id)
            elif action.startswith("blink:"):
                led_id = int(action.split(":")[1])
                result = await hardware.blink_led(led_id)
            elif action == "clear":
                result = await hardware.clear_all_leds()
            elif action == "status":
                result = await hardware.get_led_status()
            else:
                raise HTTPException(status_code=400, detail=f"Invalid LED action: {action}")
        else:
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")
        
        return {
            "test_result": "success",
            "component": component,
            "action": action,
            "response": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Hardware test failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


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
            "hardware_mode": request.app.state.hardware.__class__.__name__,
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
