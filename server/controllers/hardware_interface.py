"""
Hardware Interface Controller
Communicates with external hardware controller via HTTP for both servo and LED control
"""

import httpx
import asyncio
import json
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class HardwareInterface:
    """
    Unified interface for controlling hardware via HTTP
    Handles both servo motor and LED communication
    """
    
    def __init__(self, config_path: str = "hardware/hardware_config.json"):
        """Initialize hardware interface with configuration"""
        self.client: Optional[httpx.AsyncClient] = None
        self.config = self._load_config(config_path)
        self.base_url = self.config["hardware_controller"]["base_url"]
        self.timeout = self.config["hardware_controller"]["timeout"]
        self.endpoints = self.config["hardware_controller"]["endpoints"]
        self.compartments = self.config["compartments"]
        self.led_patterns = self.config["led_patterns"]
        
        logger.info(f"Hardware interface initialized with base URL: {self.base_url}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load hardware configuration from JSON file"""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file not found: {config_path}, using defaults")
                return self._get_default_config()
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded hardware config from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "hardware_controller": {
                "type": "http",
                "base_url": "http://localhost:8080",
                "timeout": 5,
                "endpoints": {
                    "servo_open": "/servo/open",
                    "servo_close": "/servo/close",
                    "servo_status": "/servo/status",
                    "led_on": "/led/on",
                    "led_off": "/led/off",
                    "led_blink": "/led/blink",
                    "led_clear_all": "/led/clear",
                    "led_status": "/led/status"
                }
            },
            "compartments": {
                "1": {"led_id": 1, "name": "Compartment 1"},
                "2": {"led_id": 2, "name": "Compartment 2"},
                "3": {"led_id": 3, "name": "Compartment 3"},
                "4": {"led_id": 4, "name": "Compartment 4"}
            },
            "led_patterns": {
                "success": {"duration": 1.0, "frequency": 2},
                "error": {"duration": 2.0, "frequency": 4}
            }
        }
    
    async def initialize(self):
        """Initialize HTTP client for hardware controller"""
        if self.client is None:
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout)
            )
            logger.info("HTTP client initialized for hardware controller")
    
    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
            self.client = None
            logger.info("HTTP client closed")
    
    # ========== SERVO CONTROL METHODS ==========
    
    async def open_servo(self, compartment: int) -> Dict[str, Any]:
        """
        Send open command to servo controller
        
        Args:
            compartment: Compartment number (1-4)
        
        Returns:
            Response from hardware controller
        
        Expected Request: {"compartment": 1}
        Expected Response: {"status": "success", "position": "open", "compartment": 1}
        """
        if not 1 <= compartment <= 4:
            raise ValueError(f"Invalid compartment number: {compartment}. Must be 1-4")
        
        try:
            if self.client is None:
                await self.initialize()
            
            response = await self.client.post(
                self.endpoints["servo_open"],
                json={"compartment": compartment}
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✓ Servo opened for compartment {compartment}")
            return result
            
        except httpx.TimeoutException:
            logger.error(f"✗ Timeout opening servo for compartment {compartment}")
            raise
        except httpx.HTTPError as e:
            logger.error(f"✗ HTTP error opening servo: {e}")
            raise
        except Exception as e:
            logger.error(f"✗ Unexpected error opening servo: {e}")
            raise
    
    async def close_servo(self) -> Dict[str, Any]:
        """
        Send close command to servo controller
        
        Returns:
            Response from hardware controller
        
        Expected Request: {}
        Expected Response: {"status": "success", "position": "closed"}
        """
        try:
            if self.client is None:
                await self.initialize()
            
            response = await self.client.post(self.endpoints["servo_close"])
            response.raise_for_status()
            
            result = response.json()
            logger.info("✓ Servo closed")
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"✗ HTTP error closing servo: {e}")
            raise
        except Exception as e:
            logger.error(f"✗ Unexpected error closing servo: {e}")
            raise
    
    async def get_servo_status(self) -> Dict[str, Any]:
        """
        Get current servo status from hardware controller
        
        Returns:
            Servo status information
        
        Expected Response: {
            "status": "success",
            "position": "open/closed",
            "operational": true,
            "current_compartment": 1 or null
        }
        """
        try:
            if self.client is None:
                await self.initialize()
            
            response = await self.client.get(self.endpoints["servo_status"])
            response.raise_for_status()
            
            return response.json()
            
        except httpx.HTTPError as e:
            logger.error(f"✗ HTTP error getting servo status: {e}")
            raise
        except Exception as e:
            logger.error(f"✗ Unexpected error getting servo status: {e}")
            raise
    
    # ========== LED CONTROL METHODS ==========
    
    async def turn_on_led(self, compartment: int) -> Dict[str, Any]:
        """
        Turn on LED for specific compartment
        
        Args:
            compartment: Compartment number (1-4)
        
        Returns:
            Response from hardware controller
        
        Expected Request: {"led_id": 1}
        Expected Response: {"status": "success", "led_id": 1, "state": "on"}
        """
        if not 1 <= compartment <= 4:
            raise ValueError(f"Invalid compartment number: {compartment}. Must be 1-4")
        
        try:
            if self.client is None:
                await self.initialize()
            
            led_id = self.compartments[str(compartment)]["led_id"]
            
            response = await self.client.post(
                self.endpoints["led_on"],
                json={"led_id": led_id}
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✓ LED {led_id} (compartment {compartment}) turned ON")
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"✗ HTTP error turning on LED {compartment}: {e}")
            raise
        except Exception as e:
            logger.error(f"✗ Unexpected error turning on LED {compartment}: {e}")
            raise
    
    async def turn_off_led(self, compartment: int) -> Dict[str, Any]:
        """
        Turn off LED for specific compartment
        
        Args:
            compartment: Compartment number (1-4)
        
        Returns:
            Response from hardware controller
        
        Expected Request: {"led_id": 1}
        Expected Response: {"status": "success", "led_id": 1, "state": "off"}
        """
        if not 1 <= compartment <= 4:
            raise ValueError(f"Invalid compartment number: {compartment}. Must be 1-4")
        
        try:
            if self.client is None:
                await self.initialize()
            
            led_id = self.compartments[str(compartment)]["led_id"]
            
            response = await self.client.post(
                self.endpoints["led_off"],
                json={"led_id": led_id}
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✓ LED {led_id} (compartment {compartment}) turned OFF")
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"✗ HTTP error turning off LED {compartment}: {e}")
            raise
        except Exception as e:
            logger.error(f"✗ Unexpected error turning off LED {compartment}: {e}")
            raise
    
    async def blink_led(
        self, 
        compartment: int, 
        duration: Optional[float] = None,
        frequency: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Blink LED for specific compartment
        
        Args:
            compartment: Compartment number (1-4)
            duration: How long to blink (seconds)
            frequency: Blink frequency (Hz)
        
        Returns:
            Response from hardware controller
        
        Expected Request: {"led_id": 1, "duration": 1.0, "frequency": 2}
        Expected Response: {"status": "success", "led_id": 1, "pattern": "blink"}
        """
        if not 1 <= compartment <= 4:
            raise ValueError(f"Invalid compartment number: {compartment}. Must be 1-4")
        
        # Use defaults from config if not specified
        if duration is None:
            duration = self.led_patterns["success"]["duration"]
        if frequency is None:
            frequency = self.led_patterns["success"]["frequency"]
        
        try:
            if self.client is None:
                await self.initialize()
            
            led_id = self.compartments[str(compartment)]["led_id"]
            
            response = await self.client.post(
                self.endpoints["led_blink"],
                json={
                    "led_id": led_id,
                    "duration": duration,
                    "frequency": frequency
                }
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✓ LED {led_id} (compartment {compartment}) blinking")
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"✗ HTTP error blinking LED {compartment}: {e}")
            raise
        except Exception as e:
            logger.error(f"✗ Unexpected error blinking LED {compartment}: {e}")
            raise
    
    async def clear_all_leds(self) -> Dict[str, Any]:
        """
        Turn off all LEDs
        
        Returns:
            Response from hardware controller
        
        Expected Request: {}
        Expected Response: {"status": "success", "message": "All LEDs cleared"}
        """
        try:
            if self.client is None:
                await self.initialize()
            
            response = await self.client.post(self.endpoints["led_clear_all"])
            response.raise_for_status()
            
            result = response.json()
            logger.info("✓ All LEDs cleared")
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"✗ HTTP error clearing LEDs: {e}")
            raise
        except Exception as e:
            logger.error(f"✗ Unexpected error clearing LEDs: {e}")
            raise
    
    async def get_led_status(self) -> Dict[str, Any]:
        """
        Get status of all LEDs
        
        Returns:
            LED status for all compartments
        
        Expected Response: {
            "status": "success",
            "leds": {
                "1": "on",
                "2": "off",
                "3": "off",
                "4": "off"
            }
        }
        """
        try:
            if self.client is None:
                await self.initialize()
            
            response = await self.client.get(self.endpoints["led_status"])
            response.raise_for_status()
            
            return response.json()
            
        except httpx.HTTPError as e:
            logger.error(f"✗ HTTP error getting LED status: {e}")
            raise
        except Exception as e:
            logger.error(f"✗ Unexpected error getting LED status: {e}")
            raise
    
    # ========== PATTERN METHODS ==========
    
    async def show_success_pattern(self, compartment: int) -> Dict[str, Any]:
        """Show success pattern on LED (quick blink)"""
        pattern = self.led_patterns["success"]
        return await self.blink_led(
            compartment,
            duration=pattern["duration"],
            frequency=pattern["frequency"]
        )
    
    async def show_error_pattern(self, compartment: int) -> Dict[str, Any]:
        """Show error pattern on LED (fast blink)"""
        pattern = self.led_patterns["error"]
        return await self.blink_led(
            compartment,
            duration=pattern["duration"],
            frequency=pattern["frequency"]
        )
    
    async def show_warning_pattern(self, compartment: int) -> Dict[str, Any]:
        """Show warning pattern on LED (medium blink)"""
        pattern = self.led_patterns.get("warning", self.led_patterns["success"])
        return await self.blink_led(
            compartment,
            duration=pattern["duration"],
            frequency=pattern["frequency"]
        )
    
    # ========== UTILITY METHODS ==========
    
    async def test_connection(self) -> bool:
        """
        Test connection to hardware controller
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.client is None:
                await self.initialize()
            
            # Try to get servo status as a connection test
            response = await self.client.get(
                self.endpoints["servo_status"],
                timeout=2.0
            )
            response.raise_for_status()
            
            logger.info("✓ Hardware controller connection successful")
            return True
            
        except Exception as e:
            logger.warning(f"✗ Hardware controller connection failed: {e}")
            return False
    
    def get_compartment_info(self, compartment: int) -> Dict[str, Any]:
        """Get information about a specific compartment"""
        if not 1 <= compartment <= 4:
            raise ValueError(f"Invalid compartment number: {compartment}")
        
        return self.compartments[str(compartment)]
    
    def get_all_compartments(self) -> Dict[str, Any]:
        """Get information about all compartments"""
        return self.compartments
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


# Create global instance
hardware_interface = HardwareInterface()
