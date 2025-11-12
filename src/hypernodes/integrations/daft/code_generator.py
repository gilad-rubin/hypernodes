"""Code generation for Daft pipelines.

Generates executable Python code from pipeline compilation results.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import daft
    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False
    daft = None  # type: ignore


class CodeGenerator:
    """Generates executable Daft code from pipeline compilation results.
    
    This class tracks imports, UDF definitions, and operations during pipeline
    compilation and assembles them into complete, executable Python code.
    """

    def __init__(self, debug: bool = False):
        """Initialize code generator.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
        self._reset()

    def _reset(self):
        """Reset all tracking structures."""
        self._imports: Set[Tuple[str, str]] = set()  # (module, name)
        self._udf_definitions: List[str] = []  # UDF definition code blocks
        self._operation_lines: List[str] = []  # DataFrame operations
        self._stateful_inputs: Dict[str, Any] = {}  # Stateful objects used
        self._udf_counter = 0
        self._row_id_counter = 0

    def add_import(self, module: str, name: str) -> None:
        """Track an import needed for generated code.
        
        Args:
            module: Module name (e.g., 'typing')
            name: Name to import (e.g., 'Any')
        """
        self._imports.add((module, name))

    def add_udf_definition(self, code: str) -> None:
        """Add a UDF definition to generated code.
        
        Args:
            code: Complete UDF definition as string
        """
        self._udf_definitions.append(code)

    def add_operation(self, code: str) -> None:
        """Add a DataFrame operation line.
        
        Args:
            code: Single line or block of operation code
        """
        self._operation_lines.append(code)

    def add_stateful_input(self, name: str, obj: Any) -> None:
        """Track a stateful input object.
        
        Args:
            name: Parameter name
            obj: The stateful object
        """
        self._stateful_inputs[name] = obj

    def generate_udf_name(self, base_name: str) -> str:
        """Generate unique UDF name.
        
        Args:
            base_name: Base name for UDF (e.g., 'double')
            
        Returns:
            Unique UDF name (e.g., 'double_1')
        """
        self._udf_counter += 1
        return f"{base_name}_{self._udf_counter}"

    def generate_row_id_name(self) -> str:
        """Generate unique row ID column name.
        
        Returns:
            Unique row ID column name (e.g., '__daft_row_id_1__')
        """
        self._row_id_counter += 1
        return f"__daft_row_id_{self._row_id_counter}__"

    def format_value_for_code(self, value: Any) -> str:
        """Format a value as Python code string.
        
        Returns a valid Python literal that can be used in generated code.
        Unlike reprlib.repr(), this returns the FULL representation without
        truncation, since we need executable code not debug output.
        
        Args:
            value: Value to format
            
        Returns:
            Python code string representation
        """
        if isinstance(value, str):
            return repr(value)
        elif isinstance(value, (int, float, bool)):
            return repr(value)
        elif isinstance(value, (list, tuple)):
            # Use full repr for code generation (not reprlib which truncates)
            return repr(value)
        elif value is None:
            return "None"
        else:
            # For complex objects, use placeholder
            return f"<{type(value).__name__} object>"

    def generate_code(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        output_columns: Optional[List[str]] = None,
    ) -> str:
        """Generate complete executable Python code.
        
        Args:
            inputs: Input values used in pipeline (for stateful tracking)
            output_columns: Output column names
            
        Returns:
            Complete executable Python code
        """
        lines = []
        
        # Generate header with docstring
        lines.extend(self._generate_header())
        lines.append("")
        
        # Generate imports
        lines.extend(self._generate_imports())
        lines.append("")
        
        # Generate stateful object setup
        if self._stateful_inputs:
            lines.extend(self._generate_stateful_setup())
            lines.append("")
        
        # Generate UDF definitions
        if self._udf_definitions:
            lines.append("# ==================== UDF Definitions ====================")
            lines.append("")
            for udf_def in self._udf_definitions:
                lines.append(udf_def)
                lines.append("")
        
        # Generate pipeline execution
        lines.append("")
        lines.append("# ==================== Pipeline Execution ====================")
        lines.append("")
        
        if self._operation_lines:
            for line in self._operation_lines:
                lines.append(line)
        
        return "\n".join(lines)

    def _generate_header(self) -> List[str]:
        """Generate code header with docstring and performance analysis.
        
        Returns:
            List of header lines
        """
        lines = ['"""']
        lines.append("Generated Daft code - Exact translation from HyperNodes pipeline.")
        lines.append("")
        lines.append("This code produces identical results to the HyperNodes pipeline execution.")
        lines.append("You can run this file directly to verify the translation.")
        lines.append("")
        
        # Count map operations
        map_count = sum(1 for line in self._operation_lines if "# Map over:" in line)
        if map_count > 0:
            lines.append("=" * 70)
            lines.append("PERFORMANCE ANALYSIS")
            lines.append("=" * 70)
            lines.append("")
            lines.append(f"⚠️  DETECTED {map_count} NESTED MAP OPERATIONS")
            lines.append("")
            lines.append(
                "Each map operation creates an explode → process → groupby cycle,"
            )
            lines.append(
                f"which forces data materialization. This can be {map_count}x slower than optimal."
            )
            lines.append("")
            lines.append("OPTIMIZATION STRATEGIES:")
            lines.append("")
            lines.append("1. BATCH UDFs: Use @daft.func.batch or @daft.method.batch")
            lines.append(
                "   - Processes entire Series at once (10-100x faster for ML models)"
            )
            lines.append("   - Eliminates explode/groupby overhead")
            lines.append("")
            lines.append("2. RESTRUCTURE PIPELINE: Reduce nesting")
            lines.append("   - Batch encode all passages/queries upfront")
            lines.append("   - Use vectorized operations where possible")
            lines.append("")
            lines.append("3. STATEFUL UDFs: Ensure using @daft.cls correctly")
            lines.append("   - Initialize expensive objects ONCE per worker")
            lines.append("   - Don't pass via daft.lit() - use directly!")
            lines.append("")
            lines.append("For detailed recommendations, see:")
            lines.append(
                "https://www.getdaft.io/projects/docs/en/stable/user_guide/udfs.html"
            )
            lines.append("=" * 70)
        
        lines.append('"""')
        
        return lines

    def _generate_imports(self) -> List[str]:
        """Generate import statements.
        
        Returns:
            List of import lines
        """
        lines = ["import daft"]
        
        # Group imports by module
        imports_by_module: Dict[str, List[str]] = {}
        for module, name in sorted(self._imports):
            if module not in imports_by_module:
                imports_by_module[module] = []
            imports_by_module[module].append(name)
        
        # Generate import statements
        for module, names in sorted(imports_by_module.items()):
            lines.append(f"from {module} import {', '.join(sorted(set(names)))}")
        
        return lines

    def _generate_stateful_setup(self) -> List[str]:
        """Generate stateful object initialization code.
        
        Returns:
            List of setup lines
        """
        lines = ["# ==================== Stateful Objects ===================="]
        lines.append(
            "# These objects need to be initialized before running the pipeline"
        )
        lines.append("")
        
        for name, obj in sorted(self._stateful_inputs.items()):
            lines.append(f"# {name} = <{type(obj).__name__} instance>")
            lines.append(
                "# You need to initialize this with the same configuration"
            )
        
        return lines

    def clear(self) -> None:
        """Clear all tracked state for new code generation."""
        self._reset()
