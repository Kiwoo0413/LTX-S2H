"""
nodes/griptape_compat.py
Compatibility layer for Griptape Nodes Desktop.
Provides graceful stubs when executed outside Griptape runtime (e.g. CLI tests).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Set


try:
    from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
    from griptape_nodes.exe_types.node_types import DataNode
    IS_GRIPTAPE_ENVIRONMENT = True
except ImportError:
    IS_GRIPTAPE_ENVIRONMENT = False

    class ParameterMode(Enum):
        INPUT = "INPUT"
        OUTPUT = "OUTPUT"
        PROPERTY = "PROPERTY"

    class Parameter:
        def __init__(
            self,
            name: str,
            type: str,
            default_value: Any = None,
            tooltip: str = "",
            display_name: str = "",
            allowed_modes: Optional[Set[ParameterMode]] = None,
        ) -> None:
            self.name = name
            self.type = type
            self.default_value = default_value
            self.tooltip = tooltip
            self.display_name = display_name or name
            self.allowed_modes = allowed_modes or {ParameterMode.INPUT, ParameterMode.PROPERTY}
            self.value = default_value

    class DataNode:
        def __init__(self, **kwargs: Any) -> None:
            self.parameters: Dict[str, Parameter] = {}
            self._values: Dict[str, Any] = {}

        def add_parameter(self, parameter: Parameter) -> None:
            self.parameters[parameter.name] = parameter
            self._values[parameter.name] = parameter.default_value

        def get_parameter_value(self, name: str) -> Any:
            val = self._values.get(name)
            if hasattr(val, "value"):
                return val.value
            return val

        def set_parameter_value(self, name: str, value: Any) -> None:
            self._values[name] = value
            if name in self.parameters:
                self.parameters[name].value = value

        def process(self) -> None:
            pass
