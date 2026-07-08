"""Smart-home control tool - a simulation stub.

Holds device state in memory so conversations behave consistently
("turn on the light" then "is it on?"), but drives no real hardware.
Replace ``run`` with calls to Home Assistant / MQTT / vendor APIs when
real devices arrive; the schema is designed to survive that swap.
"""

from __future__ import annotations

from typing import Any

from jarvis.tools.base import ToolError

__all__ = ["SmartHomeTool"]

_DEFAULT_DEVICES = {
    "living_room_light": "off",
    "bedroom_light": "off",
    "kitchen_light": "off",
    "thermostat": "20C",
}


class SmartHomeTool:
    """List simulated devices and set their state."""

    name = "smart_home"
    description = (
        "Control smart-home devices: list devices or set a device's "
        "state (e.g. 'on', 'off', '21C'). NOTE: this is currently a "
        "simulation stub with no real hardware attached - say so if "
        "the user asks."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "set"],
                "description": "What to do.",
            },
            "device": {"type": "string", "description": "Device name (set)."},
            "state": {"type": "string", "description": "New state (set)."},
        },
        "required": ["action"],
    }

    def __init__(self, devices: dict[str, str] | None = None) -> None:
        self._devices = dict(devices) if devices is not None else dict(_DEFAULT_DEVICES)

    async def run(self, arguments: dict[str, Any]) -> str:
        """List devices or set one device's state."""
        action = arguments.get("action")
        if action == "list":
            lines = [
                f"- {device}: {state}"
                for device, state in sorted(self._devices.items())
            ]
            return "Simulated devices:\n" + "\n".join(lines)
        if action == "set":
            device = str(arguments.get("device", "")).strip()
            state = str(arguments.get("state", "")).strip()
            if not device or not state:
                raise ToolError("'set' requires both 'device' and 'state'.")
            if device not in self._devices:
                known = ", ".join(sorted(self._devices))
                raise ToolError(f"Unknown device {device!r}. Known devices: {known}.")
            self._devices[device] = state
            return f"(simulated) Set {device} to {state}."
        raise ToolError(f"Unknown smart_home action: {action!r}")
