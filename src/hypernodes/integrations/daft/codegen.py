"""Code generation context for Daft engine.

Tracks imports, UDF definitions, and other state needed to generate executable Daft code.
"""

from typing import Any, Dict, List, Set, Tuple


class CodeGenContext:
    """Tracks state for generating Daft code."""

    def __init__(self):
        self._imports: Set[Tuple[str, str]] = set()  # (module, name)
        self._udf_definitions: List[str] = []
        self._stateful_inputs: Dict[str, Any] = {}
        self._udf_counter = 0
        self._row_id_counter = 0

    def add_import(self, module: str, name: str) -> None:
        """Add an import."""
        self._imports.add((module, name))

    def add_udf_definition(self, code: str) -> None:
        """Add a UDF definition code block."""
        self._udf_definitions.append(code)

    def add_stateful_input(self, name: str, obj: Any) -> None:
        """Track a stateful input object."""
        self._stateful_inputs[name] = obj

    def generate_udf_name(self, base_name: str) -> str:
        """Generate a unique UDF name."""
        self._udf_counter += 1
        return f"{base_name}_{self._udf_counter}"

    def generate_row_id_name(self) -> str:
        """Generate a unique row ID column name."""
        self._row_id_counter += 1
        return f"__daft_row_id_{self._row_id_counter}__"

    def format_value(self, value: Any) -> str:
        """Format a value as a Python code string."""
        if isinstance(value, str):
            return repr(value)
        elif isinstance(value, (int, float, bool)):
            return repr(value)
        elif isinstance(value, (list, tuple)):
            return repr(value)
        elif value is None:
            return "None"
        else:
            # For complex objects, we can't easily serialize them to code
            # In a real scenario, we might pickle them or assume they are available in scope
            return f"<{type(value).__name__}>"

    def generate_full_code(self, operation_lines: List[str]) -> str:
        """Generate the complete Python script."""
        lines = []
        
        # Header
        lines.append('"""Generated Daft code from HyperNodes pipeline."""')
        lines.append("")
        
        # Imports
        lines.append("import daft")
        imports_by_module: Dict[str, List[str]] = {}
        for module, name in sorted(self._imports):
            if module not in imports_by_module:
                imports_by_module[module] = []
            imports_by_module[module].append(name)
            
        for module, names in sorted(imports_by_module.items()):
            lines.append(f"from {module} import {', '.join(sorted(set(names)))}")
        lines.append("")

        # Stateful setup
        if self._stateful_inputs:
            lines.append("# Stateful Objects Setup")
            for name, obj in sorted(self._stateful_inputs.items()):
                lines.append(f"# {name} = <{type(obj).__name__} instance>")
            lines.append("")

        # UDF Definitions
        if self._udf_definitions:
            lines.append("# UDF Definitions")
            for udf in self._udf_definitions:
                lines.append(udf)
                lines.append("")
        
        # Pipeline Execution
        lines.append("# Pipeline Execution")
        lines.extend(operation_lines)
        
        return "\n".join(lines)
