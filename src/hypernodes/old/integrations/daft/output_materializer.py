"""Output materialization for Daft results."""

from typing import Any, Dict, List, Optional

try:
    import daft
    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False
    daft = None  # type: ignore


class OutputMaterializer:
    """Materializes Daft results either as Python dicts or raw DataFrames."""

    def __init__(self, mode: str = "dict"):
        """Create materializer.

        Args:
            mode: Either ``"dict"`` (default) to convert to Python objects or
                ``"daft"`` to return the collected Daft DataFrame untouched.
        """
        valid_modes = {"dict", "daft"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid output mode '{mode}'. Expected one of {valid_modes}.")
        self.mode = mode

    def materialize(
        self,
        df: Any,
        output_name: Optional[str] = None,
        squeeze: bool = True,
    ) -> Any:
        """Materialize Daft DataFrame outputs.

        Args:
            df: Daft DataFrame to materialize (already collected)
            output_name: Optional specific column to extract

        Returns:
            Dict of Python lists/scalars when mode is ``"dict"``.
            Raw Daft DataFrame when mode is ``"daft"``.

        Examples:
            >>> df = daft.from_pydict({"x": [1, 2], "y": [3, 4]})
            >>> materializer.materialize(df.collect())
            {'x': [1, 2], 'y': [3, 4]}

            >>> materializer.materialize(df.collect(), output_name="x")
            {'x': [1, 2]}
        """
        if not DAFT_AVAILABLE:
            raise ImportError("Daft is not available")

        # Select output columns if specified
        if output_name:
            df = df.select(output_name)

        if self.mode == "daft":
            return df

        result = df.to_pydict()
        if squeeze and result and all(isinstance(v, list) and len(v) == 1 for v in result.values()):
            result = {k: v[0] for k, v in result.items()}

        return result

    def materialize_map_result(self, df: Any, map_column: str) -> List[Any]:
        """Materialize the list coming out of a map operation."""
        if not DAFT_AVAILABLE:
            raise ImportError("Daft is not available")

        data = df.to_pydict()
        column = data.get(map_column, [])

        if len(column) == 1 and isinstance(column[0], list):
            return column[0]
        return column
