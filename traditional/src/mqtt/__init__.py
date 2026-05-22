"""
MQTT module for ESPMotion

Provides publish-only MQTT communication for the ESPMotion system.

"""

from .handler import MQTTHandler

__all__ = ['MQTTHandler']
