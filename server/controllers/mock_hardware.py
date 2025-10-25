"""
Mock Hardware Interface for Testing
Simulates hardware responses without actual HTTP calls
"""

import asyncio
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MockHardwareInterface:
    """
    Mock hardware interface for testing without external hardware controller
    Simulates servo and LED behavior
    """
    
    def __init__(self):
        """Initialize mock hardware with simulated state"""
        self.servo_position = "closed"
        self.current_compartment: Optional[int] = None
        self.led_states = {
            "1": "off",
            "2": "off",
            "3": "off",
            "4": "off"
        }
        self.operational = True
        logger.info("🔧 Mock hardware interface initialized")
    
    async def initialize(self):
        """Simulate initialization"""
        await asyncio.sleep(0.1)  # Simulate network delay
        logger.info("🔧 Mock HTTP client initialized")
    
    async def close(self):
        """Simulate cleanup"""
        logger.info("🔧 Mock HTTP client closed")
    
    # ========== SERVO CONTROL METHODS ==========
    
    async def open_servo(self, compartment: int) -> Dict[str, Any]:
        """Simulate opening servo"""
        if not 1 <= compartment <= 4:
            raise ValueError(f"Invalid compartment: {compartment}")
        
        await asyncio.sleep(0.2)  # Simulate servo movement
        
        self.servo_position = "open"
        self.current_compartment = compartment
        
        logger.info(f"🔧 [MOCK] Servo opened for compartment {compartment}")
        
        return {
            "status": "success",
            "position": "open",
            "compartment": compartment,
            "mock": True
        }
    
    async def close_servo(self) -> Dict[str, Any]:
        """Simulate closing servo"""
        await asyncio.sleep(0.2)  # Simulate servo movement
        
        self.servo_position = "closed"
        self.current_compartment = None
        
        logger.info("🔧 [MOCK] Servo closed")
        
        return {
            "status": "success",
            "position": "closed",
            "mock": True
        }
    
    async def get_servo_status(self) -> Dict[str, Any]:
        """Simulate getting servo status"""
        await asyncio.sleep(0.05)
        
        return {
            "status": "success",
            "position": self.servo_position,
            "operational": self.operational,
            "current_compartment": self.current_compartment,
            "mock": True
        }
    
    # ========== LED CONTROL METHODS ==========
    
    async def turn_on_led(self, compartment: int) -> Dict[str, Any]:
        """Simulate turning on LED"""
        if not 1 <= compartment <= 4:
            raise ValueError(f"Invalid compartment: {compartment}")
        
        await asyncio.sleep(0.05)
        
        self.led_states[str(compartment)] = "on"
        
        logger.info(f"🔧 [MOCK] LED {compartment} turned ON")
        
        return {
            "status": "success",
            "led_id": compartment,
            "state": "on",
            "mock": True
        }
    
    async def turn_off_led(self, compartment: int) -> Dict[str, Any]:
        """Simulate turning off LED"""
        if not 1 <= compartment <= 4:
            raise ValueError(f"Invalid compartment: {compartment}")
        
        await asyncio.sleep(0.05)
        
        self.led_states[str(compartment)] = "off"
        
        logger.info(f"🔧 [MOCK] LED {compartment} turned OFF")
        
        return {
            "status": "success",
            "led_id": compartment,
            "state": "off",
            "mock": True
        }
    
    async def blink_led(
        self,
        compartment: int,
        duration: Optional[float] = None,
        frequency: Optional[float] = None
    ) -> Dict[str, Any]:
        """Simulate blinking LED"""
        if not 1 <= compartment <= 4:
            raise ValueError(f"Invalid compartment: {compartment}")
        
        duration = duration or 1.0
        frequency = frequency or 2.0
        
        await asyncio.sleep(0.05)
        
        logger.info(f"🔧 [MOCK] LED {compartment} blinking for {duration}s at {frequency}Hz")
        
        return {
            "status": "success",
            "led_id": compartment,
            "pattern": "blink",
            "duration": duration,
            "frequency": frequency,
            "mock": True
        }
    
    async def clear_all_leds(self) -> Dict[str, Any]:
        """Simulate clearing all LEDs"""
        await asyncio.sleep(0.1)
        
        for key in self.led_states:
            self.led_states[key] = "off"
        
        logger.info("🔧 [MOCK] All LEDs cleared")
        
        return {
            "status": "success",
            "message": "All LEDs cleared",
            "mock": True
        }
    
    async def get_led_status(self) -> Dict[str, Any]:
        """Simulate getting LED status"""
        await asyncio.sleep(0.05)
        
        return {
            "status": "success",
            "leds": self.led_states.copy(),
            "mock": True
        }
    
    # ========== PATTERN METHODS ==========
    
    async def show_success_pattern(self, compartment: int) -> Dict[str, Any]:
        """Simulate success pattern"""
        return await self.blink_led(compartment, duration=1.0, frequency=2.0)
    
    async def show_error_pattern(self, compartment: int) -> Dict[str, Any]:
        """Simulate error pattern"""
        return await self.blink_led(compartment, duration=2.0, frequency=4.0)
    
    async def show_warning_pattern(self, compartment: int) -> Dict[str, Any]:
        """Simulate warning pattern"""
        return await self.blink_led(compartment, duration=1.5, frequency=3.0)
    
    # ========== UTILITY METHODS ==========
    
    async def test_connection(self) -> bool:
        """Simulate connection test (always succeeds in mock)"""
        await asyncio.sleep(0.1)
        logger.info("🔧 [MOCK] Hardware controller connection test: SUCCESS")
        return True
    
    def get_compartment_info(self, compartment: int) -> Dict[str, Any]:
        """Get mock compartment info"""
        compartments = {
            "1": {"led_id": 1, "name": "Morning Pills"},
            "2": {"led_id": 2, "name": "Afternoon Pills"},
            "3": {"led_id": 3, "name": "Evening Pills"},
            "4": {"led_id": 4, "name": "Night Pills"}
        }
        return compartments[str(compartment)]
    
    def get_all_compartments(self) -> Dict[str, Any]:
        """Get all mock compartments"""
        return {
            "1": {"led_id": 1, "name": "Morning Pills"},
            "2": {"led_id": 2, "name": "Afternoon Pills"},
            "3": {"led_id": 3, "name": "Evening Pills"},
            "4": {"led_id": 4, "name": "Night Pills"}
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


# Create global mock instance
mock_hardware_interface = MockHardwareInterface()
