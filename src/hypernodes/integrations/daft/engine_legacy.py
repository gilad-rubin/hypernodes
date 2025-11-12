"""Daft engine for HyperNodes pipelines.

This engine automatically converts HyperNodes pipelines into Daft DataFrames
using next-generation UDFs (@daft.func, @daft.cls, @daft.func.batch).

Key features:
- Automatic conversion of nodes to Daft UDFs
- Lazy evaluation and optimization
- Automatic parallelization
- Support for generators, async, and batch operations
"""

import ast
import importlib.util
import multiprocessing
import sys
import textwrap
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from hypernodes.callbacks import CallbackContext
from hypernodes.engine import Engine

try:
    import daft

    DAFT_AVAILABLE = True
except ImportError:
    DAFT_AVAILABLE = False

PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None
if PYARROW_AVAILABLE:
    import pyarrow  # type: ignore
    import pyarrow.types as pa_types  # type: ignore
else:
    pyarrow = None  # type: ignore
    pa_types = None  # type: ignore

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None

PYDANTIC_AVAILABLE = importlib.util.find_spec("pydantic") is not None
PYDANTIC_TO_PYARROW_AVAILABLE = (
    importlib.util.find_spec("pydantic_to_pyarrow") is not None
)

if PYDANTIC_AVAILABLE:
    from pydantic import BaseModel  # type: ignore
else:
    BaseModel = None  # type: ignore

if PYDANTIC_TO_PYARROW_AVAILABLE:
    from pydantic_to_pyarrow import get_pyarrow_schema  # type: ignore
else:
    get_pyarrow_schema = None  # type: ignore

if TYPE_CHECKING:
    from hypernodes.pipeline import Pipeline


def fix_script_classes_for_modal():
    """Fix all script-defined classes for Modal/distributed execution.

    Call this at the TOP of your script, immediately after imports and before
    defining any pipelines. This ensures that all classes have __module__ = "__main__"
    so they can be serialized by Modal without issues.

    Example:
        ```python
        from hypernodes import Pipeline, node
        from hypernodes.engines import DaftEngine, fix_script_classes_for_modal
        import modal

        # Fix classes BEFORE defining pipelines
        fix_script_classes_for_modal()

        # Now define your classes and pipelines
        class MyModel(BaseModel):
            ...
        ```

    This is a workaround for Modal's serialization behavior where it imports
    scripts as modules (not as __main__), causing class __module__ attributes
    to point to unimportable module names.
    """
    import sys
    import types

    classes_to_fix: Set[type] = set()

    # Find script modules (not from site-packages)
    script_module_names: Set[str] = set()

    for module_name, module in list(sys.modules.items()):
        if module is None:
            continue

        # Skip standard library
        if module_name.startswith(
            (
                "_",
                "builtins",
                "sys",
                "os",
                "typing",
                "collections",
                "functools",
                "itertools",
                "operator",
                "contextlib",
                "abc",
                "io",
                "warnings",
                "weakref",
                "copy",
                "re",
                "ast",
                "enum",
                "dataclasses",
                "logging",
                "socket",
                "ssl",
                "urllib",
                "http",
                "email",
                "xml",
                "json",
                "pickle",
                "struct",
                "array",
                "queue",
                "threading",
                "multiprocessing",
                "subprocess",
                "asyncio",
                "concurrent",
                "importlib",
                "zipimport",
                "pkgutil",
                "inspect",
                "traceback",
                "argparse",
                "configparser",
                "shutil",
                "tempfile",
                "pathlib",
                "fileinput",
                "glob",
                "fnmatch",
                "codecs",
                "locale",
                "gettext",
                "string",
                "textwrap",
                "unicodedata",
                "stringprep",
                "difflib",
                "pprint",
                "repr",
                "numbers",
                "math",
                "cmath",
                "decimal",
                "fractions",
                "random",
                "statistics",
                "datetime",
                "calendar",
                "time",
                "zoneinfo",
                "sqlite3",
                "dbm",
                "shelve",
                "marshal",
                "base64",
                "binascii",
                "quopri",
                "uu",
                "html",
                "ftplib",
                "poplib",
                "imaplib",
                "smtplib",
                "uuid",
                "socketserver",
                "selectors",
                "signal",
                "mmap",
                "ctypes",
                "platform",
                "errno",
                "getopt",
                "curses",
                "readline",
                "rlcompleter",
                "gzip",
                "bz2",
                "lzma",
                "zipfile",
                "tarfile",
                "hashlib",
                "hmac",
                "secrets",
                "crypt",
                "getpass",
                "termios",
                "tty",
                "pty",
                "fcntl",
                "resource",
                "syslog",
                "aifc",
                "wave",
                "chunk",
                "colorsys",
                "imghdr",
                "sndhdr",
                "telnetlib",
                "cgi",
                "cgitb",
                "wsgiref",
                "webbrowser",
                "ipaddress",
                "tomllib",
            )
        ):
            continue

        # Skip installed packages
        if any(
            x in module_name
            for x in [
                "site-packages",
                "dist-packages",
                ".venv",
                "hypernodes",
                "pydantic",
                "modal",
                "daft",
            ]
        ):
            continue

        # Check if it's a script (has __file__ but not from standard locations)
        if hasattr(module, "__file__") and module.__file__:
            module_file = module.__file__
            # Skip if from standard library or installed packages
            if any(
                x in module_file
                for x in [
                    "site-packages",
                    "dist-packages",
                    ".venv",
                    "/lib/python3",
                    "/lib64/python3",
                ]
            ):
                continue
            # This is likely a script file
            script_module_names.add(module_name)

    # Collect classes DEFINED in script modules
    for module_name in script_module_names:
        try:
            module = sys.modules[module_name]
            for attr_name in dir(module):
                try:
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type):
                        # Only fix if the class's __module__ matches this script module
                        class_module = getattr(attr, "__module__", None)
                        if class_module == module_name:
                            classes_to_fix.add(attr)
                except (AttributeError, TypeError):
                    pass
        except Exception:
            pass

    # Fix all classes
    for cls in classes_to_fix:
        original_module = cls.__module__

        try:
            # Ensure __main__ exists
            if "__main__" not in sys.modules:
                sys.modules["__main__"] = types.ModuleType("__main__")
            main_mod = sys.modules["__main__"]

            # Get or create original module
            if original_module in sys.modules:
                original_mod = sys.modules[original_module]
            else:
                original_mod = types.ModuleType(original_module)
                sys.modules[original_module] = original_mod

            # Change the class's __module__
            cls.__module__ = "__main__"

            # Rebind in both modules
            class_name = cls.__name__
            setattr(main_mod, class_name, cls)
            setattr(original_mod, class_name, cls)

            print(
                f"[fix_script_classes_for_modal] Fixed {original_module}.{class_name} -> __main__.{class_name}"
            )
        except Exception as e:
            print(
                f"[fix_script_classes_for_modal] Warning: Failed to fix {cls.__name__}: {e}"
            )


if TYPE_CHECKING:
    from hypernodes.pipeline import Pipeline


def _fix_annotation(annotation: Any) -> Any:
    """Fix type annotations to be serializable by value.

    Handles:
    - Simple class types: MyClass -> fixed MyClass with __module__ = "__main__"
    - Generic types: List[MyClass] -> List[fixed MyClass]
    - Union types: Union[MyClass, None] -> Union[fixed MyClass, None]

    Args:
        annotation: Type annotation to fix

    Returns:
        Fixed annotation with all class references made serializable
    """
    # Simple class type
    if isinstance(annotation, type):
        return _make_class_serializable_by_value(annotation)

    # Check for generic types (List, Dict, Optional, etc.)
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        if args:
            # Fix all type arguments recursively
            fixed_args = tuple(_fix_annotation(arg) for arg in args)
            # Reconstruct the generic type with fixed arguments
            # Use typing module's internal mechanism
            try:
                if origin is Union:
                    # Special handling for Union (includes Optional)
                    from typing import Union as UnionType

                    return UnionType[fixed_args]
                elif origin is list or origin is List:
                    return List[fixed_args[0]] if len(fixed_args) == 1 else List
                elif origin is dict or origin is Dict:
                    if len(fixed_args) >= 2:
                        return Dict[fixed_args[0], fixed_args[1]]
                    else:
                        return Dict
                else:
                    # Try generic reconstruction
                    return origin[fixed_args]
            except Exception:
                # If reconstruction fails, return the fixed args as a tuple
                # Better than failing completely
                return annotation

    # For other annotations (str, int, Any, etc.), return as-is
    return annotation


def _make_serializable_by_value(obj):
    """Force cloudpickle to serialize function/class by value instead of by reference.

    This is critical for Modal/distributed execution where the original module
    may not exist in worker processes. By setting __module__ = "__main__",
    cloudpickle will serialize the entire object bytecode instead of just
    storing an import path.

    Args:
        obj: Function or class to make serializable by value

    Returns:
        A new function/class object with __module__ = "__main__"
    """
    # Detect if this is a class type
    if isinstance(obj, type):
        return _make_class_serializable_by_value(obj)

    # Otherwise treat as function
    import types

    try:
        # Create a NEW function object with the same bytecode but different module
        # This is more robust than just modifying __module__ in place
        new_func = types.FunctionType(
            obj.__code__,  # Same bytecode
            obj.__globals__,  # Same global namespace
            obj.__name__,  # Same name
            obj.__defaults__,  # Same default arguments
            obj.__closure__,  # Same closure variables
        )

        # Set module to __main__ to force by-value serialization
        new_func.__module__ = "__main__"
        new_func.__qualname__ = obj.__name__

        # Copy other important attributes
        new_func.__dict__.update(obj.__dict__)

        # Fix annotations - replace any class types with serializable versions
        if obj.__annotations__:
            fixed_annotations = {}

            # IMPORTANT: If annotations are strings (from __future__ import annotations),
            # we need to evaluate them first using get_type_hints() to get actual types,
            # THEN fix those types, THEN store them back as the actual fixed types
            # (not strings) so Daft sees the fixed classes
            try:
                # Try to get evaluated type hints (converts strings to actual types)
                type_hints = get_type_hints(obj)
                use_type_hints = True
            except Exception:
                # If get_type_hints fails, fall back to raw annotations
                type_hints = obj.__annotations__
                use_type_hints = False

            for key, annotation in type_hints.items():
                fixed_annotation = _fix_annotation(annotation)
                fixed_annotations[key] = fixed_annotation
                # Debug: check if we fixed a class from a script
                if annotation != fixed_annotation and use_type_hints:
                    orig_mod = (
                        getattr(annotation, "__module__", "unknown")
                        if hasattr(annotation, "__module__")
                        else "N/A"
                    )
                    fixed_mod = (
                        getattr(fixed_annotation, "__module__", "unknown")
                        if hasattr(fixed_annotation, "__module__")
                        else "N/A"
                    )
                    print(
                        f"[FIX_ANNOTATION] {obj.__name__}.{key}: module {orig_mod} -> {fixed_mod}"
                    )

            new_func.__annotations__ = fixed_annotations
        else:
            new_func.__annotations__ = {}

        new_func.__doc__ = obj.__doc__

        return new_func
    except (AttributeError, TypeError, ValueError):
        # Fallback: try to modify in place
        try:
            obj.__module__ = "__main__"
            obj.__qualname__ = obj.__name__
        except (AttributeError, TypeError):
            pass
        return obj


def _make_class_serializable_by_value(cls: type) -> type:
    """Create a new class type with __module__ = "__main__" for by-value pickling.

    This allows classes defined in scripts or inner functions to be pickled
    by cloudpickle without requiring the original module to be importable.

    Args:
        cls: Class type to transform

    Returns:
        A new class type with the same attributes but __module__ = "__main__"
    """
    # Skip built-in types
    if cls.__module__.startswith("builtins"):
        return cls

    # If already __main__, no need to transform
    if cls.__module__ in ("__main__", "__mp_main__"):
        return cls

    try:
        # Create a new class type with the same bases and namespace
        # Use type() to create the new class dynamically
        class_dict = {}

        # Copy class attributes (excluding special attributes that can't be copied)
        exclude_attrs = {
            "__dict__",
            "__weakref__",
            "__module__",
            "__qualname__",
            "__mro__",
            "__base__",
            "__bases__",
        }

        for attr_name in dir(cls):
            if attr_name in exclude_attrs:
                continue

            try:
                attr_value = getattr(cls, attr_name)
                # Only copy if it's defined directly on this class, not inherited
                if attr_name in cls.__dict__:
                    class_dict[attr_name] = attr_value
            except (AttributeError, TypeError):
                # Some attributes can't be accessed via getattr
                continue

        # Preserve __slots__ if present
        if hasattr(cls, "__slots__"):
            class_dict["__slots__"] = cls.__slots__

        original_module_name = cls.__module__

        # Create new class with same name and bases
        new_cls = type(cls.__name__, cls.__bases__, class_dict)

        # Set module to __main__ to force by-value serialization
        new_cls.__module__ = "__main__"
        new_cls.__qualname__ = cls.__name__

        # Copy docstring
        if cls.__doc__:
            new_cls.__doc__ = cls.__doc__

        # CRITICAL: Fix the class's own __annotations__ (Pydantic field types)
        # This is essential for Pydantic models where field annotations might
        # reference other classes from the same script
        if hasattr(cls, "__annotations__") and cls.__annotations__:
            try:
                # Get actual type hints (evaluates string annotations)
                type_hints = get_type_hints(cls)
                fixed_annotations = {}
                for key, annotation in type_hints.items():
                    fixed_annotations[key] = _fix_annotation(annotation)
                new_cls.__annotations__ = fixed_annotations
            except Exception:
                # If fixing fails, copy original annotations
                new_cls.__annotations__ = cls.__annotations__.copy()

        # Replace original class binding inside its module so future references
        # (e.g., globals in script files) resolve to the serializable version.
        module_obj = sys.modules.get(original_module_name)
        if module_obj is not None:
            try:
                setattr(module_obj, cls.__name__, new_cls)
            except Exception:
                pass

        return new_cls

    except (TypeError, AttributeError):
        # If class creation fails, try in-place modification as fallback
        try:
            cls.__module__ = "__main__"
            cls.__qualname__ = cls.__name__
        except (AttributeError, TypeError):
            pass
        return cls


_FIXED_INSTANCES: Set[int] = set()


def _fix_instance_class(instance: Any) -> Any:
    """Fix a class instance to have a serializable-by-value class.

    Modifies the instance's __class__ attribute to point to a version
    of the class with __module__ = "__main__", enabling cloudpickle to
    serialize it by value instead of by reference.

    Args:
        instance: Object instance to fix

    Returns:
        The same instance, potentially with modified __class__
    """
    # Track instances we've already fixed to avoid redundant work
    instance_id = id(instance)
    if instance_id in _FIXED_INSTANCES:
        return instance

    # Get the instance's class
    cls = instance.__class__

    # Skip built-in types
    if cls.__module__.startswith("builtins"):
        return instance

    # Skip if already __main__
    if cls.__module__ in ("__main__", "__mp_main__"):
        _FIXED_INSTANCES.add(instance_id)
        return instance

    # Check if class needs by-value serialization
    try:
        spec = importlib.util.find_spec(cls.__module__)
        if spec is not None:
            # Module is importable in THIS process, but check if it's a script
            # Scripts won't be importable in worker processes (Modal, etc.)
            # Check if the module origin is a script (not in site-packages/stdlib)
            if spec.origin:
                import os

                origin_path = os.path.abspath(spec.origin)
                # If it's in site-packages or stdlib, it's a real package
                if "site-packages" in origin_path or "lib/python" in origin_path:
                    # Real installed package, cloudpickle can handle by reference
                    return instance
                # Otherwise, it's likely a script that needs by-value serialization
            else:
                # No origin info, assume it's safe to serialize by reference
                return instance
    except (ImportError, ValueError, AttributeError):
        # Module not importable, needs by-value serialization
        pass

    try:
        # Create serializable version of the class
        new_cls = _make_class_serializable_by_value(cls)

        # For Pydantic models with frozen=True, we can't modify __class__ directly
        # Check if this is a frozen Pydantic model
        is_frozen_pydantic = False
        if PYDANTIC_AVAILABLE and BaseModel is not None:
            try:
                if isinstance(instance, BaseModel):
                    config = getattr(cls, "model_config", {})
                    if isinstance(config, dict):
                        is_frozen_pydantic = config.get("frozen", False)
            except (TypeError, AttributeError):
                pass

        if is_frozen_pydantic:
            # For frozen Pydantic models, we need to be more careful
            # Some fields might not serialize well (e.g., numpy arrays)
            try:
                # First try: use model_copy with the new class
                # This preserves all fields including arbitrary types
                new_instance = instance.model_copy(deep=True)
                object.__setattr__(new_instance, "__class__", new_cls)
                _FIXED_INSTANCES.add(id(new_instance))
                return new_instance
            except Exception:
                # Second try: use model_dump() and recreate
                try:
                    model_data = instance.model_dump()
                    new_instance = new_cls(**model_data)
                    _FIXED_INSTANCES.add(id(new_instance))
                    return new_instance
                except Exception:
                    # Third try: Create empty instance and copy __dict__
                    try:
                        new_instance = object.__new__(new_cls)
                        for k, v in instance.__dict__.items():
                            object.__setattr__(new_instance, k, v)
                        _FIXED_INSTANCES.add(id(new_instance))
                        return new_instance
                    except Exception:
                        # If all else fails, try direct __class__ assignment
                        pass

        # Modify the instance's class in place
        object.__setattr__(instance, "__class__", new_cls)
        _FIXED_INSTANCES.add(instance_id)

    except (TypeError, AttributeError):
        # Some objects don't allow __class__ modification
        # This is okay - cloudpickle might still handle them
        pass

    return instance


def _is_builtin_type(obj: Any) -> bool:
    """Check if object is a builtin type that doesn't need serialization fixing.

    Args:
        obj: Object to check

    Returns:
        True if object is a builtin type (str, int, etc.) that doesn't need fixing
    """
    if obj is None:
        return True

    if isinstance(obj, (str, int, float, bool, bytes, bytearray, complex, type(None))):
        return True

    # Check module
    try:
        obj_module = type(obj).__module__
        if obj_module in ("builtins", "__builtin__", "typing"):
            return True
    except (AttributeError, TypeError):
        pass

    return False


def _ensure_module_pickled_by_value(module_name: str) -> None:
    """Register a module to be pickled by value using cloudpickle.

    This tells cloudpickle to serialize the entire module by value instead of
    just storing an import reference. This is critical for script modules that
    won't exist in worker environments.

    Args:
        module_name: Name of the module to register (e.g., 'test_script')
    """
    # Skip standard library and builtin modules
    if module_name in ("builtins", "__builtin__", "typing", "__main__", "__mp_main__"):
        return

    try:
        import sys

        import cloudpickle

        if module_name in sys.modules:
            module = sys.modules[module_name]
            try:
                # Register module for by-value pickling
                cloudpickle.register_pickle_by_value(module)
            except Exception:
                # cloudpickle might not support this or module might not be registerable
                pass
    except ImportError:
        # cloudpickle not available
        pass
    except Exception:
        # Other errors, just continue
        pass


def _fix_object_deeply(
    obj: Any,
    visited: Optional[Set[int]] = None,
    max_depth: int = 10,
    current_depth: int = 0,
) -> Any:
    """Recursively fix all custom class instances in an object's attributes.

    This is the comprehensive solution that walks through an object's entire
    attribute tree and fixes any custom classes it finds. This catches:
    - Nested stateful objects (e.g., reranker._encoder)
    - Objects in lists/tuples/dicts
    - Closure variables

    Args:
        obj: Object to fix
        visited: Set of object IDs already visited (prevents cycles)
        max_depth: Maximum recursion depth to prevent infinite loops
        current_depth: Current recursion depth

    Returns:
        Fixed object with all nested custom classes having __module__ = "__main__"
    """
    # Safety: prevent infinite recursion
    if current_depth >= max_depth:
        return obj

    # Safety: prevent cycles
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return obj
    visited.add(obj_id)

    # Skip builtin types
    if _is_builtin_type(obj):
        return obj

    # Fix the object's class first
    try:
        obj = _fix_instance_class(obj)
    except Exception:
        # If fixing fails, continue with the object as-is
        pass

    # Recursively fix attributes
    if hasattr(obj, "__dict__"):
        try:
            for attr_name, attr_value in list(obj.__dict__.items()):
                try:
                    # Fix class references
                    if isinstance(attr_value, type):
                        obj.__dict__[attr_name] = _make_class_serializable_by_value(
                            attr_value
                        )

                    # Fix custom class instances
                    elif not _is_builtin_type(attr_value):
                        if isinstance(attr_value, (list, tuple)):
                            # Fix items in collections
                            fixed_items = [
                                _fix_object_deeply(
                                    item, visited, max_depth, current_depth + 1
                                )
                                if not _is_builtin_type(item)
                                else item
                                for item in attr_value
                            ]
                            obj.__dict__[attr_name] = type(attr_value)(fixed_items)

                        elif isinstance(attr_value, dict):
                            # Fix values in dicts
                            fixed_dict = {
                                k: _fix_object_deeply(
                                    v, visited, max_depth, current_depth + 1
                                )
                                if not _is_builtin_type(v)
                                else v
                                for k, v in attr_value.items()
                            }
                            obj.__dict__[attr_name] = fixed_dict

                        elif isinstance(attr_value, set):
                            # Fix items in sets (if they're custom objects)
                            try:
                                fixed_set = {
                                    _fix_object_deeply(
                                        item, visited, max_depth, current_depth + 1
                                    )
                                    if not _is_builtin_type(item)
                                    else item
                                    for item in attr_value
                                }
                                obj.__dict__[attr_name] = fixed_set
                            except TypeError:
                                # Set items might not be fixable, skip
                                pass

                        else:
                            # Fix nested object
                            obj.__dict__[attr_name] = _fix_object_deeply(
                                attr_value, visited, max_depth, current_depth + 1
                            )

                except (AttributeError, TypeError, ValueError):
                    # Some attributes can't be modified, skip them
                    continue

        except Exception:
            # If attribute iteration fails, continue
            pass

    return obj


def _register_fixed_class_in_sys_modules(original_cls: type, fixed_cls: type) -> None:
    """Register fixed class in sys.modules so it can be found by Daft workers.

    Args:
        original_cls: Original class type
        fixed_cls: Fixed class with __module__ = "__main__"
    """
    try:
        import sys
        import types

        # Ensure __main__ module exists
        if "__main__" not in sys.modules:
            sys.modules["__main__"] = types.ModuleType("__main__")

        main_module = sys.modules["__main__"]

        # Add the fixed class to __main__
        if not hasattr(main_module, original_cls.__name__):
            setattr(main_module, original_cls.__name__, fixed_cls)

    except Exception:
        # If registration fails, just continue
        pass


def _prepare_stateful_value_for_daft(value: Any, debug: bool = False) -> Any:
    """Comprehensive preparation of a stateful value for Daft serialization.

    This is the main entry point that applies all serialization fixes:
    1. Deep recursive fixing of nested objects
    2. Module registration with cloudpickle
    3. sys.modules registration for class lookup

    Args:
        value: Stateful value to prepare
        debug: Whether to print debug messages

    Returns:
        Fixed value ready for Daft serialization
    """
    if value is None or _is_builtin_type(value):
        return value

    cls = value.__class__
    original_module = cls.__module__

    # Skip if already __main__
    if original_module in ("__main__", "__mp_main__"):
        return value

    # Skip builtin modules
    if original_module in ("builtins", "__builtin__", "typing"):
        return value

    try:
        # Step 1: Register original module with cloudpickle for by-value serialization
        _ensure_module_pickled_by_value(original_module)

        # Step 2: Deep recursive fixing - this is the comprehensive fix
        fixed_value = _fix_object_deeply(value)

        # Step 3: Register fixed class in sys.modules
        _register_fixed_class_in_sys_modules(cls, fixed_value.__class__)

        try:
            setattr(fixed_value, "__hn_fixed_by_value__", True)
        except Exception:
            pass

        if debug and fixed_value.__class__.__module__ != original_module:
            print(
                f"[DaftEngine] Deep-fixed {original_module}.{cls.__name__} "
                f"-> {fixed_value.__class__.__module__}.{fixed_value.__class__.__name__}"
            )

        return fixed_value

    except Exception as exc:
        if debug:
            print(f"[DaftEngine] Warning: Failed to deep-fix {cls.__name__}: {exc}")
        return value


def _fix_output_tree(value: Any, debug: bool = False) -> Any:
    """Recursively prepare returned values for by-value serialization."""

    if isinstance(value, dict):
        return {k: _fix_output_tree(v, debug=debug) for k, v in value.items()}

    if isinstance(value, list):
        return [_fix_output_tree(v, debug=debug) for v in value]

    if isinstance(value, tuple):
        return tuple(_fix_output_tree(v, debug=debug) for v in value)

    if isinstance(value, set):
        return {_fix_output_tree(v, debug=debug) for v in value}

    # Fix individual values
    fixed = _prepare_stateful_value_for_daft(value, debug=debug)

    if debug and fixed is not value:
        orig_cls = value.__class__
        fixed_cls = fixed.__class__
        if orig_cls.__module__ != fixed_cls.__module__:
            print(
                f"[_fix_output_tree] Fixed {orig_cls.__module__}.{orig_cls.__name__} "
                f"-> {fixed_cls.__module__}.{fixed_cls.__name__}"
            )

    return fixed


_REGISTERED_BY_VALUE_CLASSES: Set[type] = set()
_REGISTER_PICKLE_BY_VALUE: Optional[Callable[[type], None]] = None


def _get_register_pickle_by_value() -> Optional[Callable[[type], None]]:
    """Lazily fetch daft/cloudpickle's register_pickle_by_value helper."""

    global _REGISTER_PICKLE_BY_VALUE

    if _REGISTER_PICKLE_BY_VALUE is not None:
        return _REGISTER_PICKLE_BY_VALUE

    candidate = None
    try:
        from daft.pickle import cloudpickle as daft_cloudpickle  # type: ignore

        candidate = getattr(daft_cloudpickle, "register_pickle_by_value", None)
    except ImportError:
        try:
            import cloudpickle as cloudpickle_module  # type: ignore

            candidate = getattr(cloudpickle_module, "register_pickle_by_value", None)
        except ImportError:
            candidate = None

    _REGISTER_PICKLE_BY_VALUE = candidate
    return _REGISTER_PICKLE_BY_VALUE


class DaftEngine(Engine):
    """Daft engine that converts HyperNodes pipelines to Daft DataFrames.

    This engine translates HyperNodes pipelines into Daft operations:
    - Nodes become @daft.func UDFs
    - Map operations become DataFrame operations
    - Pipelines are converted to lazy DataFrame transformations

    **Concurrency Control for Stateful Objects**:

    When using stateful objects (e.g., ML models, tokenizers) that are identified
    as needing @daft.cls wrapping, the engine automatically applies sensible defaults:

    - ``use_process=False``: Uses threads instead of processes (avoids PyTorch/CUDA fork issues)

    You can optionally override these defaults by adding hints to your classes:

    .. code-block:: python

        class MyModel:
            # Optional: Tell DaftEngine this is stateful
            __daft_hint__ = "@daft.cls"
            __daft_stateful__ = True

            # Optional: Limit concurrent instances (useful for GPU memory)
            __daft_max_concurrency__ = 1

            # Optional: Override use_process default
            __daft_use_process__ = False

            # Optional: Request GPUs
            __daft_gpus__ = 1

    These hints are **completely optional** - the engine works out-of-the-box with
    PyTorch/HuggingFace models without any configuration.

    Args:
        collect: Whether to automatically collect results (default: True)
        show_plan: Whether to print the execution plan (default: False)
        debug: Whether to enable debug mode (default: False)
        python_return_strategy: How to materialize Python outputs when
            ``return_format='python'``. Options:
            - "auto": prefer Arrow conversion when available, fallback to pydict
            - "pydict": previous behavior using ``to_pydict()``
            - "arrow": force Arrow-based conversion (requires pyarrow)
            - "pandas": materialize via ``to_pandas()`` (requires pandas)
        force_spawn_method: Whether to force multiprocessing spawn method (default: True)

    Example:
        >>> from hypernodes import node, Pipeline
        >>> from hypernodes.executors import DaftEngine
        >>>
        >>> @node(output_name="result")
        >>> def add_one(x: int) -> int:
        >>>     return x + 1
        >>>
        >>> pipeline = Pipeline(nodes=[add_one], engine=DaftEngine())
        >>> result = pipeline.run(inputs={"x": 5})
        >>> # result == {"result": 6}
    """

    PYTHON_STRATEGIES = {"auto", "pydict", "arrow", "pandas"}

    def __init__(
        self,
        collect: bool = True,
        show_plan: bool = False,
        debug: bool = False,
        python_return_strategy: str = "auto",
        force_spawn_method: bool = True,
        code_generation_mode: bool = False,
    ):
        if not DAFT_AVAILABLE:
            raise ImportError(
                "Daft is not installed. Install it with: pip install daft"
            )
        self.collect = collect
        self.show_plan = show_plan
        self.debug = debug
        if python_return_strategy not in self.PYTHON_STRATEGIES:
            raise ValueError(
                f"python_return_strategy must be one of "
                f"{sorted(self.PYTHON_STRATEGIES)}, got '{python_return_strategy}'"
            )
        if python_return_strategy in {"arrow"} and not PYARROW_AVAILABLE:
            raise RuntimeError(
                "python_return_strategy='arrow' requires pyarrow to be installed"
            )
        if python_return_strategy == "pandas" and not PANDAS_AVAILABLE:
            raise RuntimeError(
                "python_return_strategy='pandas' requires pandas to be installed"
            )
        self.python_return_strategy = python_return_strategy

        # Code generation mode
        self.code_generation_mode = code_generation_mode
        if code_generation_mode:
            self._init_code_generation()

        # Configure multiprocessing for PyTorch/CUDA compatibility
        if force_spawn_method and not code_generation_mode:
            self._configure_multiprocessing_for_pytorch()

        # CRITICAL: Fix all script-defined classes at initialization time
        # This ensures that any classes from unimportable scripts get their
        # __module__ fixed BEFORE they're used in pipeline operations
        if not code_generation_mode:
            self._fix_script_classes_at_init()

    def _fix_script_classes_at_init(self) -> None:
        """Fix all script-defined classes at engine initialization.

        This scans sys.modules (especially __main__) for classes from unimportable
        modules and fixes them by changing their __module__ to "__main__".

        This is critical for Modal/distributed execution where script classes
        won't be importable on workers.
        """
        import importlib.util
        import sys
        import types

        classes_to_fix: Set[type] = set()

        # Check __main__ and the calling module
        modules_to_scan = []
        if "__main__" in sys.modules:
            modules_to_scan.append(sys.modules["__main__"])

        # Also scan any modules that look like scripts (not in standard locations)
        for module_name, module in list(sys.modules.items()):
            if module is None:
                continue
            # Skip standard library and installed packages
            if module_name.startswith(("_", "builtins", "sys", "os", "typing")):
                continue
            if any(
                x in module_name
                for x in ["site-packages", "dist-packages", ".venv", "hypernodes."]
            ):
                continue
            # Check if module is from a script (not importable via standard mechanisms)
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    # Module not importable - likely a script
                    modules_to_scan.append(module)
            except (ImportError, ValueError, AttributeError):
                # Module not importable - likely a script
                modules_to_scan.append(module)

        # Collect all classes from these modules
        for module in modules_to_scan:
            try:
                for attr_name in dir(module):
                    try:
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type):
                            # Check if this class needs fixing
                            if self._class_requires_by_value_serialization(attr):
                                classes_to_fix.add(attr)
                    except (AttributeError, TypeError):
                        pass
            except Exception:
                pass

        # Fix each class
        for cls in classes_to_fix:
            original_module = cls.__module__

            if original_module in ("__main__", "__mp_main__"):
                continue

            try:
                # Get or create the original module object
                if original_module in sys.modules:
                    original_module_obj = sys.modules[original_module]
                else:
                    original_module_obj = types.ModuleType(original_module)
                    sys.modules[original_module] = original_module_obj

                # Ensure __main__ exists
                if "__main__" not in sys.modules:
                    sys.modules["__main__"] = types.ModuleType("__main__")
                main_module = sys.modules["__main__"]

                # Change the class's __module__ to __main__
                cls.__module__ = "__main__"
                try:
                    setattr(cls, "__hn_from_script__", True)
                except Exception:
                    pass

                # Rebind in both modules
                class_name = cls.__name__
                setattr(main_module, class_name, cls)
                setattr(original_module_obj, class_name, cls)

                if self.debug:
                    print(
                        f"[DaftEngine.__init__] Fixed {original_module}.{class_name} -> __main__.{class_name}"
                    )
            except Exception as e:
                if self.debug:
                    print(
                        f"[DaftEngine.__init__] Warning: Failed to fix {cls.__name__}: {e}"
                    )

    def _configure_multiprocessing_for_pytorch(self):
        """Configure multiprocessing to avoid PyTorch/CUDA fork issues.

        PyTorch and CUDA have known issues with the 'fork' multiprocessing method,
        which can cause segmentation faults. This method sets the start method to
        'spawn' to avoid these issues.

        See: https://pytorch.org/docs/stable/notes/multiprocessing.html
        """
        current_method = multiprocessing.get_start_method(allow_none=True)

        if current_method == "spawn":
            # Already configured correctly
            return

        if current_method is not None and current_method != "spawn":
            # Method already set to something else (fork/forkserver)
            warnings.warn(
                f"Multiprocessing start method is already set to '{current_method}'. "
                f"For PyTorch/CUDA compatibility, DaftEngine recommends 'spawn'. "
                f"This may cause segmentation faults with PyTorch models. "
                f"To fix: call multiprocessing.set_start_method('spawn') before "
                f"importing torch or creating DaftEngine, or set force_spawn_method=False "
                f"to disable this warning.",
                RuntimeWarning,
                stacklevel=3,
            )
            return

        # Set spawn method
        try:
            multiprocessing.set_start_method("spawn", force=False)
            if self.debug:
                print(
                    "[DaftEngine] Set multiprocessing start method to 'spawn' for PyTorch compatibility"
                )
        except RuntimeError:
            # Already set by another module
            warnings.warn(
                "Could not set multiprocessing start method to 'spawn'. "
                "If using PyTorch/CUDA, you may experience segmentation faults. "
                "Call multiprocessing.set_start_method('spawn') at the top of your script.",
                RuntimeWarning,
                stacklevel=3,
            )

    def run(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        output_name: Union[str, List[str], None] = None,
    ) -> Dict[str, Any]:
        """Execute a pipeline by converting it to a Daft DataFrame.

        Args:
            pipeline: The pipeline to execute
            inputs: Dictionary of input values
            output_name: Optional output name(s) to compute

        Returns:
            Dictionary containing the pipeline outputs
        """
        # CRITICAL: Fix all script-defined classes BEFORE any Daft operations
        # This ensures that when Daft creates Expression objects, any class references
        # have __module__ = "__main__" instead of unimportable script names
        self._fix_all_script_classes(pipeline, inputs)

        # Register all script modules with cloudpickle for by-value pickling
        self._register_all_script_modules(inputs)

        # Store inputs and output_name for code generation
        if self.code_generation_mode:
            self._actual_inputs = dict(inputs)
            self._actual_output_name = output_name

        df_inputs, stateful_inputs, stateful_literal_keys = (
            self._split_inputs_for_dataframe(inputs)
        )
        df_data = {k: [v] for k, v in df_inputs.items()}
        if not df_data:
            df_data["__hn_stateful_placeholder__"] = [0]

        # Create a single-row DataFrame from non-stateful inputs only
        df = daft.from_pydict(df_data)

        # Convert pipeline to DataFrame operations
        df = self._convert_pipeline_to_daft(
            pipeline, df, set(df_inputs.keys()), stateful_inputs
        )

        # In code generation mode, generate the from_pydict call AFTER conversion
        # so we know which inputs are stateful
        if self.code_generation_mode:
            # Insert dataframe creation at the beginning of generated code
            df_creation = []
            self._stateful_literal_input_keys = stateful_literal_keys
            self._stateful_actual_inputs = {
                key: value
                for key, value in stateful_inputs.items()
                if key in stateful_literal_keys
            }
            self._generate_dataframe_creation_lines(inputs, df_creation)
            # Insert at the beginning
            self.generated_code = df_creation + self.generated_code

        if self.show_plan and not self.code_generation_mode:
            print("Daft Execution Plan:")
            print(df.explain())

        requested_outputs = self._resolve_requested_outputs(pipeline, output_name)
        if not requested_outputs:
            return {}

        # In code generation mode, generate the select and collect calls and return
        if self.code_generation_mode:
            self._generate_output_collection(requested_outputs)
            # Return empty dict in code generation mode (no actual execution)
            return {}

        df = self._select_output_columns(df, requested_outputs)

        try:
            if self.collect:
                df = df.collect()

            # Extract requested results (single row)
            result_dict = df.to_pydict()
            row_values = {}
            for key in requested_outputs:
                column = result_dict.get(key)
                if column:
                    row_values[key] = self._convert_output_value(column[0])

            return row_values
        except Exception as exc:
            fallback = self._fallback_to_hypernodes(
                pipeline, inputs, output_name, _ctx, exc
            )
            if fallback is not None:
                return fallback
            raise

    def _fallback_to_hypernodes(
        self,
        pipeline: "Pipeline",
        inputs: Dict[str, Any],
        output_name: Union[str, List[str], None],
        _ctx: Optional[CallbackContext],
        exc: Exception,
    ) -> Optional[Dict[str, Any]]:
        """Fallback execution path that uses HypernodesEngine when Daft fails."""

        try:
            from hypernodes.engine import HypernodesEngine
        except Exception:
            return None

        if getattr(self, "debug", False):
            print(
                "[DaftEngine] Falling back to HypernodesEngine due to Daft error:"
                f" {exc}"
            )

        original_engine = pipeline.engine
        try:
            pipeline.engine = HypernodesEngine()
            return pipeline.engine.run(
                pipeline, inputs, output_name=output_name, _ctx=_ctx
            )
        finally:
            pipeline.engine = original_engine

    def map(
        self,
        pipeline: "Pipeline",
        items: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        output_name: Union[str, List[str], None] = None,
        _ctx: Optional[CallbackContext] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a pipeline over multiple items using Daft.

        Args:
            pipeline: The pipeline to execute
            items: List of input dictionaries (one per item)
            inputs: Shared inputs for all items
            output_name: Optional output name(s) to compute
            _ctx: Internal callback context (not used in Daft engine)

        Returns:
            List of output dictionaries (one per item)
        """
        if not items:
            # Handle empty case: return an empty list of per-item dicts
            return []

        # Merge items with shared inputs
        # Each item gets its own row with shared inputs broadcast
        data_dict = {}

        # Add varying parameters from items
        for key in items[0].keys():
            data_dict[key] = [item[key] for item in items]

        shared_inputs, stateful_inputs, _ = self._split_inputs_for_dataframe(inputs)

        # Add fixed parameters from shared inputs (excluding stateful ones)
        for key, value in shared_inputs.items():
            if key not in data_dict:
                data_dict[key] = [value] * len(items)

        stateful_constant_cols: List[str] = []
        for key, values in data_dict.items():
            if not values:
                continue
            first_value = values[0]
            is_constant = True
            for value in values[1:]:
                equal = value is first_value
                if not equal:
                    try:
                        equal = value == first_value
                    except Exception:
                        equal = False
                if not equal:
                    is_constant = False
                    break
            if is_constant:
                stateful_inputs.setdefault(key, first_value)
                if getattr(self, "debug", False):
                    print(f"[DaftEngine] Detected constant column '{key}'")
                if self._has_daft_cls_hint(
                    first_value
                ) and self._pipeline_accepts_stateful_param(pipeline, key, first_value):
                    stateful_constant_cols.append(key)

        for key in stateful_constant_cols:
            data_dict.pop(key, None)

        outputs = self._map_dataframe(
            pipeline,
            data_dict,
            output_name,
            initial_stateful_inputs=stateful_inputs,
            return_format="python",
        )

        # Convert dict-of-lists into list-of-dicts for Pipeline.map
        if not outputs:
            return []

        num_rows = len(next(iter(outputs.values())))
        results_list: List[Dict[str, Any]] = []
        for i in range(num_rows):
            row = {key: outputs[key][i] for key in outputs}
            results_list.append(row)

        return results_list

    def map_columnar(
        self,
        pipeline: "Pipeline",
        varying_inputs: Dict[str, List[Any]],
        fixed_inputs: Dict[str, Any],
        output_name: Union[str, List[str], None],
        return_format: str = "python",
        _ctx: Optional[CallbackContext] = None,
    ) -> Optional[Any]:
        """Columnar fast-path for Pipeline.map when map_mode='zip'.

        Returns:
            Engine-native output honoring ``return_format`` if handled,
            otherwise ``None`` to signal fallback to Python execution.
        """
        if not varying_inputs:
            return {}

        # Determine number of rows
        first_key = next(iter(varying_inputs.keys()))
        num_rows = len(varying_inputs[first_key])
        if num_rows == 0:
            return {}

        # Build columnar data dict
        data_dict: Dict[str, List[Any]] = {
            key: list(values) for key, values in varying_inputs.items()
        }

        fixed_columns, stateful_inputs, _ = self._split_inputs_for_dataframe(
            fixed_inputs
        )

        for key, value in fixed_columns.items():
            if isinstance(value, list):
                if len(value) != num_rows:
                    raise ValueError(
                        f"Fixed parameter '{key}' list length {len(value)} "
                        f"does not match varying inputs length {num_rows}"
                    )
                data_dict[key] = list(value)
            else:
                data_dict[key] = [value] * num_rows

        return self._map_dataframe(
            pipeline,
            data_dict,
            output_name,
            initial_stateful_inputs=stateful_inputs,
            return_format=return_format,
        )

    def _convert_pipeline_to_daft(
        self,
        pipeline: "Pipeline",
        df: "daft.DataFrame",
        input_keys: Set[str],
        stateful_inputs: Dict[str, Any],
    ) -> "daft.DataFrame":
        """Convert a HyperNodes pipeline to Daft DataFrame operations.

        Args:
            pipeline: The pipeline to convert
            df: The input DataFrame
            input_keys: Set of input column names

        Returns:
            DataFrame with all pipeline transformations applied
        """
        if stateful_inputs and getattr(self, "debug", False):
            print(
                f"[DaftEngine] Converting pipeline '{getattr(pipeline, 'name', pipeline)}' "
                f"with stateful inputs: {list(stateful_inputs.keys())}"
            )

        # Track available columns
        available = set(input_keys)

        # Process nodes in topological order
        for node in pipeline.nodes:
            # Check if this is a PipelineNode (wraps a pipeline)
            if hasattr(node, "pipeline") and hasattr(node, "input_mapping"):
                # PipelineNode - handle input/output mapping
                df = self._convert_pipeline_node_to_daft(
                    node, df, available, stateful_inputs
                )
                # Update available columns with mapped output names
                output_name = node.output_name
                if isinstance(output_name, tuple):
                    available.update(output_name)
                else:
                    available.add(output_name)
            # Check if this is a direct Pipeline (has 'nodes' attribute)
            elif hasattr(node, "nodes") and hasattr(node, "run"):
                # Nested pipeline - recursively convert
                df = self._convert_pipeline_to_daft(
                    node, df, available, stateful_inputs
                )
                # Update available columns with nested pipeline outputs
                available.update(self._get_output_names(node))
            elif hasattr(node, "func"):
                # Regular node - convert to UDF
                df = self._convert_node_to_daft(node, df, available, stateful_inputs)
                # Update available columns
                if hasattr(node, "output_name"):
                    available.add(node.output_name)
            else:
                # Unknown node type - skip or raise error
                raise ValueError(f"Unknown node type: {type(node)}")

        return df

    def _convert_pipeline_node_to_daft(
        self,
        pipeline_node: Any,
        df: "daft.DataFrame",
        available: Set[str],
        stateful_inputs: Dict[str, Any],
    ) -> "daft.DataFrame":
        """Convert a PipelineNode (created with .as_node()) to Daft operations.

        Supports both map_over and non-map_over PipelineNodes:
        - Without map_over: Simple input/output mapping
        - With map_over: Uses explode() -> transform -> groupby().agg(list())

        Args:
            pipeline_node: The PipelineNode to convert
            df: The input DataFrame
            available: Set of available column names

        Returns:
            DataFrame with the PipelineNode's transformations applied
        """
        # Check if this PipelineNode uses map_over
        if hasattr(pipeline_node, "map_over") and pipeline_node.map_over:
            return self._convert_mapped_pipeline_node(
                pipeline_node, df, available, stateful_inputs
            )

        # Get the inner pipeline and mappings
        inner_pipeline = pipeline_node.pipeline
        input_mapping = pipeline_node.input_mapping or {}
        output_mapping = pipeline_node.output_mapping or {}

        # For PipelineNodes without map_over, we can handle input/output mapping
        # by renaming columns before and after the inner pipeline execution

        # Apply input mapping: rename outer columns to inner names
        inner_stateful_inputs = dict(stateful_inputs)

        if input_mapping:
            # Create column aliases for the inner pipeline
            rename_exprs = []
            inner_available = set()
            mapped_stateful = dict(stateful_inputs)
            for outer_name, inner_name in input_mapping.items():
                if outer_name in stateful_inputs:
                    mapped_stateful[inner_name] = stateful_inputs[outer_name]
            inner_stateful_inputs = mapped_stateful

            for outer_name, inner_name in input_mapping.items():
                if outer_name in available:
                    # Rename: outer_name -> inner_name
                    rename_exprs.append(df[outer_name].alias(inner_name))
                    inner_available.add(inner_name)

            # Keep other available columns as-is
            for col_name in available:
                if col_name not in input_mapping:
                    rename_exprs.append(df[col_name])
                    inner_available.add(col_name)

            # Create new DataFrame with renamed columns
            if rename_exprs:
                df = df.select(*rename_exprs)
        else:
            inner_available = available.copy()

        inner_available.update(inner_stateful_inputs.keys())

        # Convert the inner pipeline (modifies df in place)
        df = self._convert_pipeline_to_daft(
            inner_pipeline, df, inner_available, inner_stateful_inputs
        )

        # Apply output mapping: rename inner outputs to outer names
        if output_mapping:
            inner_outputs = self._get_output_names(inner_pipeline)

            # Rename outputs according to mapping
            rename_exprs = []

            # Keep all existing columns
            for col_name in df.column_names:
                if col_name in inner_outputs:
                    # This is an output - apply mapping
                    outer_name = output_mapping.get(col_name, col_name)
                    if outer_name != col_name:
                        rename_exprs.append(df[col_name].alias(outer_name))
                    else:
                        rename_exprs.append(df[col_name])
                else:
                    # Not an output - keep as-is
                    rename_exprs.append(df[col_name])

            df = df.select(*rename_exprs)

        return df

    def _convert_mapped_pipeline_node(
        self,
        pipeline_node: Any,
        df: "daft.DataFrame",
        available: Set[str],
        stateful_inputs: Dict[str, Any],
    ) -> "daft.DataFrame":
        """Convert a PipelineNode with map_over to Daft operations.

        Strategy:
        1. Add unique row ID for grouping later
        2. Explode the list column into multiple rows
        3. Apply input mapping (rename columns)
        4. Run inner pipeline on exploded rows
        5. Apply output mapping
        6. Group by row ID and collect results back into lists

        Args:
            pipeline_node: The PipelineNode with map_over set
            df: The input DataFrame
            available: Set of available column names

        Returns:
            DataFrame with mapped transformations applied
        """
        import daft

        # Get configuration
        inner_pipeline = pipeline_node.pipeline
        input_mapping = pipeline_node.input_mapping or {}
        output_mapping = pipeline_node.output_mapping or {}
        map_over = pipeline_node.map_over

        # Validate map_over is a string (single column)
        if isinstance(map_over, list):
            if len(map_over) != 1:
                raise ValueError(
                    f"DaftBackend only supports map_over with a single column. "
                    f"Got map_over={map_over}"
                )
            map_over_col = map_over[0]
        else:
            map_over_col = map_over

        # The map_over_col should be in input_mapping keys (outer name)
        if map_over_col not in input_mapping:
            raise ValueError(
                f"map_over column '{map_over_col}' not found in input_mapping. "
                f"Expected one of: {list(input_mapping.keys())}"
            )

        # Step 1: Add unique row ID for grouping later
        if self.code_generation_mode:
            row_id_col = self._generate_row_id_name()
        else:
            row_id_col = "__daft_row_id__"
        original_mapped_col = f"__original_{map_over_col}__"

        # Code generation mode
        if self.code_generation_mode:
            self._generate_operation_code("# Map over: " + map_over_col)
            self._generate_operation_code(
                'df = df.with_column("' + row_id_col + '", daft.lit(0))'
            )
            self._generate_operation_code(
                'df = df.with_column("'
                + original_mapped_col
                + '", df["'
                + map_over_col
                + '"])'
            )
            self._generate_operation_code(
                'df = df.explode(daft.col("' + map_over_col + '"))'
            )
        else:
            # For DaftBackend with map_over, we need to materialize to add row IDs
            # This is a limitation when dealing with Python object types
            # Collect just the column names and row count to minimize issues
            try:
                # Try to add row ID without materializing
                df = df.with_column(row_id_col, daft.lit(0))
                # Try to copy the mapped column
                df = df.with_column(original_mapped_col, df[map_over_col])
            except Exception:
                # Fallback: materialize to add row IDs
                df_collected = df.to_pydict()
                num_rows = len(list(df_collected.values())[0])
                df_collected[row_id_col] = list(range(num_rows))
                df_collected[original_mapped_col] = df_collected[map_over_col]
                df = daft.from_pydict(df_collected)

            # Step 2: Explode the list column
            df_exploded = df.explode(daft.col(map_over_col))

        if self.code_generation_mode:
            df_exploded = df  # Not actually modifying in code gen mode

        # Step 3: Apply input mapping - rename outer columns to inner names
        if not self.code_generation_mode:
            rename_exprs = []
            inner_available = set()

            for outer_name, inner_name in input_mapping.items():
                if outer_name in available or outer_name == map_over_col:
                    # Rename: outer_name -> inner_name
                    rename_exprs.append(df_exploded[outer_name].alias(inner_name))
                    inner_available.add(inner_name)

            # Keep other available columns (including row_id_col)
            for col_name in df_exploded.column_names:
                if col_name not in input_mapping and col_name != map_over_col:
                    rename_exprs.append(df_exploded[col_name])
                    if col_name != row_id_col:
                        inner_available.add(col_name)

            df_exploded = df_exploded.select(*rename_exprs)
        else:
            # In code generation mode, emit alias operations so generated code has renamed columns
            inner_available = set()
            for outer_name, inner_name in input_mapping.items():
                if outer_name in available or outer_name == map_over_col:
                    inner_available.add(inner_name)
                    self._generate_operation_code(
                        f'df = df.with_column("{inner_name}", df["{outer_name}"])'
                    )
            # Also include non-mapped columns
            for col in available:
                if col not in input_mapping and col != map_over_col:
                    inner_available.add(col)

        # Step 4: Run inner pipeline on exploded DataFrame
        inner_stateful_inputs = dict(stateful_inputs)
        mapped_stateful = {}
        for outer_name, inner_name in input_mapping.items():
            if outer_name in stateful_inputs:
                mapped_stateful[inner_name] = stateful_inputs[outer_name]

        inner_stateful_inputs.update(mapped_stateful)

        inner_available.update(inner_stateful_inputs.keys())

        # Capture columns that exist before the inner pipeline runs
        # These need to be preserved in the groupby aggregation
        pre_map_columns = set(available) - {map_over_col}

        df_transformed = self._convert_pipeline_to_daft(
            inner_pipeline, df_exploded, inner_available, inner_stateful_inputs
        )

        # Step 5: Apply output mapping (if any)
        inner_outputs = self._get_output_names(inner_pipeline)

        if output_mapping:
            rename_exprs = []
            for col_name in df_transformed.column_names:
                if col_name in inner_outputs:
                    outer_name = output_mapping.get(col_name, col_name)
                    rename_exprs.append(df_transformed[col_name].alias(outer_name))
                else:
                    rename_exprs.append(df_transformed[col_name])
            df_transformed = df_transformed.select(*rename_exprs)

        # Update output names after mapping
        final_output_names = [output_mapping.get(name, name) for name in inner_outputs]

        if self.code_generation_mode and output_mapping:
            for inner_name, outer_name in output_mapping.items():
                if inner_name == outer_name:
                    continue
                self._generate_operation_code(
                    f'df = df.with_column("{outer_name}", df["{inner_name}"])'
                )

        # Check if original_mapped_col survived through the inner pipeline
        has_original_col = original_mapped_col in df_transformed.column_names

        # Get list of columns to keep (non-output, non-row-id, non-original-mapped)
        # Include both:
        # 1. Columns from the inner pipeline (post-map columns)
        # 2. Columns that existed before the map (pre-map columns like indexes)
        if self.code_generation_mode:
            # In code gen mode, compute what columns WOULD exist after inner pipeline
            post_map_columns = set(df_exploded.column_names) | inner_outputs
        else:
            # In runtime mode, use actual DataFrame columns
            post_map_columns = set(df_transformed.column_names)

        all_columns_to_consider = pre_map_columns | post_map_columns

        keep_cols = [
            col
            for col in all_columns_to_consider
            if col not in final_output_names
            and col != row_id_col
            and col != original_mapped_col
            # In runtime mode, only keep cols that actually exist in the transformed DataFrame
            and (self.code_generation_mode or col in df_transformed.column_names)
        ]

        # Step 6: Group by row ID and collect results into lists
        if self.code_generation_mode:
            agg_code_parts = []
            for output_name in final_output_names:
                agg_code_parts.append(
                    f'daft.col("{output_name}").list_agg().alias("{output_name}")'
                )
            for col_name in keep_cols:
                agg_code_parts.append(
                    f'daft.col("{col_name}").any_value().alias("{col_name}")'
                )
            if map_over_col:
                agg_code_parts.append(
                    f'daft.col("{original_mapped_col}").any_value().alias("{map_over_col}")'
                )

            agg_code = ", ".join(agg_code_parts)
            self._generate_operation_code(
                'df = df.groupby(daft.col("' + row_id_col + '")).agg(' + agg_code + ")"
            )

            # Remove row_id column and backup column (if present)
            select_cols = final_output_names + keep_cols
            if map_over_col:
                select_cols.append(map_over_col)
            select_args = ", ".join([f'df["{col}"]' for col in select_cols])
            if select_args:
                self._generate_operation_code(f"df = df.select({select_args})")
            else:
                self._generate_operation_code("df = df.select()")

            return df_transformed

        # Group by row_id and aggregate outputs into lists
        df_grouped = df_transformed.groupby(daft.col(row_id_col))

        # Collect each output into a list using list_agg()
        # Build list of aggregation expressions
        agg_exprs = []

        for output_name in final_output_names:
            # Use list_agg() (preferred) or agg_list() (deprecated)
            try:
                agg_exprs.append(daft.col(output_name).list_agg().alias(output_name))
            except AttributeError:
                # Fallback to deprecated agg_list()
                agg_exprs.append(daft.col(output_name).agg_list().alias(output_name))

        # Also get an arbitrary value (typically first) of non-output columns
        for col_name in keep_cols:
            agg_exprs.append(daft.col(col_name).any_value().alias(col_name))

        # Restore the original mapped column (before exploding) if it survived
        if has_original_col:
            agg_exprs.append(
                daft.col(original_mapped_col).any_value().alias(map_over_col)
            )

        df_result = df_grouped.agg(*agg_exprs)

        # Step 7: Remove row_id column and original_mapped_col backup
        select_exprs = [
            df_result[col]
            for col in df_result.column_names
            if col != row_id_col and col != original_mapped_col
        ]
        df_result = df_result.select(*select_exprs)

        return df_result

    def _convert_node_to_daft(
        self,
        node: Any,
        df: "daft.DataFrame",
        available: Set[str],
        stateful_inputs: Dict[str, Any],
    ) -> "daft.DataFrame":
        """Convert a single node to a Daft UDF and apply it.

        Args:
            node: The node to convert
            df: The input DataFrame
            available: Set of available column names

        Returns:
            DataFrame with the node's transformation applied
        """
        # Get node function and output name
        func = node.func
        output_name = node.output_name

        # Get node parameters that are available either as columns or stateful inputs
        params = [p for p in node.root_args if p in available or p in stateful_inputs]

        if stateful_inputs and getattr(self, "debug", False):
            print(
                f"[DaftEngine] Node '{getattr(node, 'output_name', node)}' "
                f"stateful candidates: {list(stateful_inputs.keys())}"
            )

        # Check if return type is Pydantic and wrap if needed
        try:
            type_hints = get_type_hints(func)
            # Fix ALL type hints to ensure custom classes are serializable
            # This is critical for Daft's internal Expression objects
            fixed_type_hints = {}
            for hint_name, hint_type in type_hints.items():
                fixed_type_hints[hint_name] = _fix_annotation(hint_type)
            type_hints = fixed_type_hints
            return_type = type_hints.get("return", None)
        except Exception:
            type_hints = {}
            return_type = None

        stateful_values = self._resolve_stateful_params(
            params, stateful_inputs, type_hints
        )
        dynamic_params = [p for p in params if p not in stateful_values]

        if getattr(self, "debug", False):
            print(
                f"[DaftEngine] Node '{output_name}' dynamic_params={dynamic_params} "
                f"stateful_params={list(stateful_values.keys())}"
            )

        # Track stateful input names for code generation
        if self.code_generation_mode:
            for param_name in stateful_values.keys():
                self._stateful_input_names.add(param_name)

        try:
            # Check parameter types for Pydantic models (for input conversion)
            # This dict maps param_name -> (container_type, pydantic_type)
            # container_type is 'single' or 'list'
            param_pydantic_types = {}
            if PYDANTIC_AVAILABLE and BaseModel is not None:
                for param in params:
                    param_type = type_hints.get(param)
                    if param_type is not None:
                        try:
                            # Check if it's a direct Pydantic model
                            if isinstance(param_type, type) and issubclass(
                                param_type, BaseModel
                            ):
                                param_pydantic_types[param] = ("single", param_type)
                            else:
                                # Check if it's List[PydanticModel]
                                origin = get_origin(param_type)
                                if origin in (list, List):
                                    args = get_args(param_type)
                                    if args and len(args) > 0:
                                        elem_type = args[0]
                                        if isinstance(elem_type, type) and issubclass(
                                            elem_type, BaseModel
                                        ):
                                            param_pydantic_types[param] = (
                                                "list",
                                                elem_type,
                                            )
                        except TypeError:
                            pass

            # Check if return type is a Pydantic model
            is_pydantic_return = False
            if PYDANTIC_AVAILABLE and BaseModel is not None and return_type is not None:
                try:
                    if isinstance(return_type, type) and issubclass(
                        return_type, BaseModel
                    ):
                        is_pydantic_return = True
                except TypeError:
                    pass

            # Wrap function to handle both input conversion and output serialization
            # Apply serialization fix to original function FIRST
            original_func = _make_serializable_by_value(
                func
            )  # Always store original for code generation

            # IMPORTANT: Fix Pydantic model classes for Modal/distributed execution
            # The Pydantic classes themselves need __module__ = "__main__" so they can be pickled
            fixed_param_pydantic_types = {}
            for param_name, (
                container_type,
                pydantic_type,
            ) in param_pydantic_types.items():
                fixed_pydantic_type = _make_serializable_by_value(pydantic_type)
                fixed_param_pydantic_types[param_name] = (
                    container_type,
                    fixed_pydantic_type,
                )

            # Update param_pydantic_types with fixed versions
            param_pydantic_types = fixed_param_pydantic_types

            # ALWAYS wrap functions to apply _fix_output_tree for Modal/distributed execution
            # This ensures ALL outputs get their module names fixed, not just Pydantic models
            debug_mode = self.debug

            if param_pydantic_types or is_pydantic_return:
                # Full wrapper with Pydantic conversion + output fixing
                def wrapped_func(*args, **kwargs):
                    try:
                        # Convert dict inputs to Pydantic models
                        converted_args = []
                        for arg, param_name in zip(args, params):
                            if param_name in param_pydantic_types:
                                container_type, pydantic_type = param_pydantic_types[
                                    param_name
                                ]

                                if container_type == "single":
                                    # Convert single dict to Pydantic model
                                    if isinstance(arg, dict):
                                        try:
                                            arg = pydantic_type(**arg)
                                        except Exception as e:
                                            if debug_mode:
                                                print(
                                                    f"Error converting dict to {pydantic_type.__name__}: {e}"
                                                )
                                            # If conversion fails, pass as-is
                                            pass
                                    # Handle tuple/list representations (from PyArrow structs)
                                    elif isinstance(
                                        arg, (tuple, list)
                                    ) and not isinstance(arg, pydantic_type):
                                        try:
                                            # Try to convert tuple to dict first
                                            if all(
                                                isinstance(item, tuple)
                                                and len(item) == 2
                                                for item in arg
                                            ):
                                                arg_dict = {k: v for k, v in arg}
                                                arg = pydantic_type(**arg_dict)
                                        except Exception as e:
                                            if debug_mode:
                                                print(
                                                    f"Error converting tuple to {pydantic_type.__name__}: {e}"
                                                )
                                            # If conversion fails, pass as-is
                                            pass

                                elif container_type == "list":
                                    # Convert list of dicts to list of Pydantic models
                                    if isinstance(arg, list):
                                        converted_list = []
                                        for item in arg:
                                            if isinstance(item, dict):
                                                try:
                                                    converted_list.append(
                                                        pydantic_type(**item)
                                                    )
                                                except Exception as e:
                                                    if debug_mode:
                                                        print(
                                                            f"Error converting list item to {pydantic_type.__name__}: {e}"
                                                        )
                                                    # If conversion fails, use as-is
                                                    converted_list.append(item)
                                            elif isinstance(item, pydantic_type):
                                                # Already a Pydantic model
                                                converted_list.append(item)
                                            else:
                                                # Try to convert other formats
                                                try:
                                                    # Handle tuple format
                                                    if isinstance(
                                                        item, (tuple, list)
                                                    ) and all(
                                                        isinstance(x, tuple)
                                                        and len(x) == 2
                                                        for x in item
                                                    ):
                                                        item_dict = {
                                                            k: v for k, v in item
                                                        }
                                                        converted_list.append(
                                                            pydantic_type(**item_dict)
                                                        )
                                                    else:
                                                        converted_list.append(item)
                                                except Exception as e:
                                                    if debug_mode:
                                                        print(
                                                            f"Error converting tuple item to {pydantic_type.__name__}: {e}"
                                                        )
                                                    converted_list.append(item)
                                        arg = converted_list

                            converted_args.append(arg)

                        # Convert kwargs if needed
                        converted_kwargs = {}
                        for k, v in kwargs.items():
                            if k in param_pydantic_types:
                                container_type, pydantic_type = param_pydantic_types[k]

                                if container_type == "single" and isinstance(v, dict):
                                    try:
                                        v = pydantic_type(**v)
                                    except Exception as e:
                                        if debug_mode:
                                            print(
                                                f"Error converting kwarg {k} to {pydantic_type.__name__}: {e}"
                                            )
                                        pass
                                elif container_type == "list" and isinstance(v, list):
                                    converted_list = []
                                    for item in v:
                                        if isinstance(item, dict):
                                            try:
                                                converted_list.append(
                                                    pydantic_type(**item)
                                                )
                                            except Exception as e:
                                                if debug_mode:
                                                    print(
                                                        f"Error converting kwarg list item to {pydantic_type.__name__}: {e}"
                                                    )
                                                converted_list.append(item)
                                        else:
                                            converted_list.append(item)
                                    v = converted_list

                            converted_kwargs[k] = v

                        # Call original function
                        result = original_func(*converted_args, **converted_kwargs)

                        # Ensure outputs are safe for cross-process serialization.
                        if self.debug:
                            print(f"[DaftEngine] Fixing output from {output_name}")
                        return _fix_output_tree(result, debug=self.debug)
                    except Exception as e:
                        # Log the error with context
                        import traceback

                        print(f"\n{'=' * 60}")
                        print(f"ERROR in wrapped function for node: {node.output_name}")
                        print(f"Function: {original_func.__name__}")
                        print(f"Error: {e}")
                        print(f"{'=' * 60}")
                        traceback.print_exc()
                        print(f"{'=' * 60}\n")
                        raise

                # Force by-value serialization for Modal/distributed execution
                func = _make_serializable_by_value(wrapped_func)
            else:
                # Simple wrapper that only applies _fix_output_tree (no Pydantic conversion)
                def wrapped_func(*args, **kwargs):
                    result = original_func(*args, **kwargs)
                    # Always fix outputs for Modal/distributed execution
                    if debug_mode:
                        print(f"[DaftEngine] Fixing output from {output_name}")
                    return _fix_output_tree(result, debug=debug_mode)

                # Force by-value serialization for Modal/distributed execution
                func = _make_serializable_by_value(wrapped_func)

            # Infer Daft dtype
            inferred_dtype = None
            if return_type is not None:
                inferred_dtype = self._infer_return_dtype(return_type)

            # Code generation mode: Generate UDF code instead of executing
            if self.code_generation_mode:
                udf_name = self._generate_udf_code(
                    original_func, output_name, params, stateful_values, inferred_dtype
                )

                # Generate with_column code
                col_args = ", ".join([f'df["{p}"]' for p in dynamic_params])
                self._generate_operation_code(
                    f'df = df.with_column("{output_name}", {udf_name}({col_args}))'
                )

                # Return unchanged df (not actually executing)
                return df

            # Fallback: execute stateful nodes in Python when objects come from non-importable modules
            if stateful_values and self._should_execute_stateful_in_python(
                stateful_values
            ):
                return self._execute_stateful_node_in_python(
                    df,
                    dynamic_params,
                    stateful_values,
                    func,
                    output_name,
                    inferred_dtype,
                )

            # Normal execution mode
            if stateful_values:
                daft_func = self._build_stateful_udf(
                    func, params, stateful_values, inferred_dtype
                )
            elif inferred_dtype is not None:
                # Force by-value serialization for Modal/distributed execution
                serializable_func = _make_serializable_by_value(func)
                daft_func = daft.func(serializable_func, return_dtype=inferred_dtype)
            else:
                # No usable hint, try automatic inference
                # Force by-value serialization for Modal/distributed execution
                serializable_func = _make_serializable_by_value(func)
                daft_func = daft.func(serializable_func)
        except Exception:
            # If anything fails, fall back to Python object storage
            if self.code_generation_mode:
                # In code generation mode, still generate code
                # Use original_func if available, otherwise func
                func_to_gen = original_func if "original_func" in locals() else func
                udf_name = self._generate_udf_code(
                    func_to_gen,
                    output_name,
                    params,
                    stateful_values,
                    daft.DataType.python(),
                )
                col_args = ", ".join([f'df["{p}"]' for p in dynamic_params])
                self._generate_operation_code(
                    f'df = df.with_column("{output_name}", {udf_name}({col_args}))'
                )
                return df
            else:
                # Force by-value serialization for Modal/distributed execution
                serializable_func = _make_serializable_by_value(func)
                daft_func = daft.func(
                    serializable_func, return_dtype=daft.DataType.python()
                )

        # Apply the UDF to the DataFrame
        # Build column expressions for parameters
        col_exprs = [df[param] for param in dynamic_params]
        if getattr(self, "debug", False) and dynamic_params:
            expr_descriptions = [str(expr) for expr in col_exprs]
            print(
                f"[DaftEngine] Node '{output_name}' column expressions: {expr_descriptions}"
            )

        # Special case: If no dynamic params, call the function directly and use daft.lit()
        # Daft UDFs require at least one column input, so we can't use them for zero-parameter functions
        if not dynamic_params and not stateful_values:
            # Node with no inputs - call directly and store result
            result = func()
            # Fix output for Modal/distributed execution
            if self.debug:
                print(f"[DaftEngine] Fixing direct call output from {output_name}")
            result = _fix_output_tree(result, debug=self.debug)

            expr = daft.lit(result)
            if inferred_dtype is not None:
                expr = expr.cast(inferred_dtype)
            df = df.with_column(output_name, expr)
        elif not dynamic_params and stateful_values:
            # Node with only stateful inputs - call with stateful values only
            # We need to wrap the result in daft.lit() since there are no column inputs
            result = func(**stateful_values)
            # Fix output for Modal/distributed execution
            if self.debug:
                print(
                    f"[DaftEngine] Fixing stateful-only call output from {output_name}"
                )
            result = _fix_output_tree(result, debug=self.debug)

            expr = daft.lit(result)
            if inferred_dtype is not None:
                expr = expr.cast(inferred_dtype)
            df = df.with_column(output_name, expr)
        else:
            # Normal case: apply UDF to DataFrame columns
            df = df.with_column(output_name, daft_func(*col_exprs))

        return df

    def _get_output_names(self, pipeline: "Pipeline") -> Set[str]:
        """Get all output names from a pipeline, including nested pipelines.

        Args:
            pipeline: The pipeline to analyze

        Returns:
            Set of output names
        """
        outputs = set()
        for node in pipeline.nodes:
            # Check if this is a PipelineNode (wraps a pipeline)
            if hasattr(node, "pipeline") and hasattr(node, "input_mapping"):
                # PipelineNode - get outputs from wrapped pipeline, then apply output_mapping
                inner_outputs = self._get_output_names(node.pipeline)

                # Apply output mapping if present
                if hasattr(node, "output_mapping") and node.output_mapping:
                    output_mapping = node.output_mapping
                    # Map inner names to outer names
                    mapped_outputs = set()
                    for inner_name in inner_outputs:
                        outer_name = output_mapping.get(inner_name, inner_name)
                        mapped_outputs.add(outer_name)
                    outputs.update(mapped_outputs)
                else:
                    outputs.update(inner_outputs)
            # Check if this is a direct Pipeline (has 'nodes' attribute)
            elif hasattr(node, "nodes") and hasattr(node, "run"):
                # Nested pipeline - recursively get outputs
                outputs.update(self._get_output_names(node))
            elif hasattr(node, "output_name"):
                # Regular node - add its output
                output = node.output_name
                if isinstance(output, tuple):
                    outputs.update(output)
                else:
                    outputs.add(output)
        return outputs

    def _convert_output_value(self, value: Any) -> Any:
        """Recursively convert Daft/PyArrow values into standard Python types."""

        if PYARROW_AVAILABLE and pyarrow is not None:
            if isinstance(value, pyarrow.Scalar):
                return self._convert_output_value(value.as_py())
            if isinstance(value, pyarrow.Array):
                return [self._convert_output_value(item) for item in value.to_pylist()]
            if isinstance(value, pyarrow.ChunkedArray):
                return [self._convert_output_value(item) for item in value.to_pylist()]

        if isinstance(value, dict):
            return {k: self._convert_output_value(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._convert_output_value(v) for v in value]

        if isinstance(value, tuple):
            if all(isinstance(item, tuple) and len(item) == 2 for item in value):
                return {k: self._convert_output_value(v) for k, v in value}
            return [self._convert_output_value(v) for v in value]

        return value

    def _resolve_requested_outputs(
        self, pipeline: "Pipeline", output_name: Union[str, List[str], None]
    ) -> List[str]:
        """Determine which outputs should be materialized/collected."""

        ordered_outputs = self._ordered_output_names(pipeline)
        available = set(ordered_outputs)

        if output_name is None:
            return ordered_outputs

        if isinstance(output_name, str):
            requested = [output_name]
        else:
            # Preserve user order while removing duplicates
            seen = set()
            requested = []
            for name in output_name:
                if name not in seen:
                    requested.append(name)
                    seen.add(name)

        missing = [name for name in requested if name not in available]
        if missing:
            raise ValueError(
                f"Requested output(s) {missing} not found in pipeline outputs {ordered_outputs}"
            )
        return requested

    def _ordered_output_names(self, pipeline: "Pipeline") -> List[str]:
        """Get output names in deterministic pipeline order."""

        ordered: List[str] = []
        for node in pipeline.nodes:
            if hasattr(node, "pipeline") and hasattr(node, "input_mapping"):
                inner_outputs = self._ordered_output_names(node.pipeline)
                output_mapping = getattr(node, "output_mapping", None) or {}
                for name in inner_outputs:
                    mapped = output_mapping.get(name, name)
                    if mapped not in ordered:
                        ordered.append(mapped)
            elif hasattr(node, "nodes") and hasattr(node, "run"):
                for name in self._ordered_output_names(node):
                    if name not in ordered:
                        ordered.append(name)
            elif hasattr(node, "output_name"):
                output_name = node.output_name
                if isinstance(output_name, tuple):
                    for name in output_name:
                        if name not in ordered:
                            ordered.append(name)
                else:
                    if output_name not in ordered:
                        ordered.append(output_name)
        return ordered

    def _select_output_columns(
        self, df: "daft.DataFrame", columns: List[str]
    ) -> "daft.DataFrame":
        """Return DataFrame narrowed to requested columns only."""

        if not columns:
            return df

        missing = [col for col in columns if col not in df.column_names]
        if missing:
            raise ValueError(f"Missing expected output columns: {missing}")

        return df.select(*(df[col] for col in columns))

    def _map_dataframe(
        self,
        pipeline: "Pipeline",
        data_dict: Dict[str, List[Any]],
        output_name: Union[str, List[str], None],
        initial_stateful_inputs: Optional[Dict[str, Any]] = None,
        return_format: str = "python",
    ) -> Any:
        """Execute pipeline using a prepared columnar dict."""

        stateful_inputs = dict(initial_stateful_inputs or {})

        # Detect constant columns that should be treated as stateful
        for key, values in data_dict.items():
            if not values:
                continue
            first_value = values[0]
            is_constant = True
            for value in values[1:]:
                equal = value is first_value
                if not equal:
                    try:
                        equal = value == first_value
                    except Exception:
                        equal = False
                if not equal:
                    is_constant = False
                    break
            if is_constant and key not in stateful_inputs:
                stateful_inputs[key] = first_value
                if getattr(self, "debug", False):
                    print(f"[DaftEngine] Detected constant column '{key}'")

        # Remove stateful-only columns from DataFrame inputs
        for key in list(stateful_inputs.keys()):
            value = stateful_inputs[key]
            if (
                key in data_dict
                and self._should_capture_stateful_input(value)
                and self._pipeline_accepts_stateful_param(pipeline, key, value)
            ):
                data_dict.pop(key, None)

        if not data_dict:
            if return_format == "python":
                return {}
            raise ValueError(
                "No varying inputs available for columnar execution; "
                f"return_format='{return_format}' cannot be produced."
            )

        df = daft.from_pydict(data_dict)

        input_keys = set(data_dict.keys())
        requested_outputs = self._resolve_requested_outputs(pipeline, output_name)
        if not requested_outputs:
            return {}

        df = self._convert_pipeline_to_daft(pipeline, df, input_keys, stateful_inputs)
        df = self._select_output_columns(df, requested_outputs)

        if self.show_plan:
            print("Daft Execution Plan:")
            print(df.explain())

        materialized = df.collect() if self.collect else df

        if return_format == "daft":
            return materialized

        if return_format == "arrow":
            if not PYARROW_AVAILABLE:
                raise RuntimeError(
                    "return_format='arrow' requires pyarrow to be installed"
                )
            return materialized.to_arrow()

        if return_format != "python":
            raise ValueError(f"Unsupported return_format '{return_format}'")

        return self._materialize_python_outputs(materialized, requested_outputs)

    def _materialize_python_outputs(
        self, materialized: "daft.DataFrame", requested_outputs: List[str]
    ) -> Dict[str, List[Any]]:
        """Convert Daft DataFrame into python outputs according to strategy."""

        if not requested_outputs:
            return {}

        strategy = self.python_return_strategy
        if strategy == "auto":
            return self._convert_via_pydict(materialized, requested_outputs)
        if strategy == "pydict":
            return self._convert_via_pydict(materialized, requested_outputs)
        if strategy == "arrow":
            if not PYARROW_AVAILABLE:
                raise RuntimeError(
                    "python_return_strategy='arrow' requires pyarrow to be installed"
                )
            return self._convert_via_arrow(materialized, requested_outputs)
        if strategy == "pandas":
            if not PANDAS_AVAILABLE:
                raise RuntimeError(
                    "python_return_strategy='pandas' requires pandas to be installed"
                )
            return self._convert_via_pandas(materialized, requested_outputs)

        raise ValueError(f"Unsupported python_return_strategy '{strategy}'")

    def _convert_via_pydict(
        self, materialized: "daft.DataFrame", requested_outputs: List[str]
    ) -> Dict[str, List[Any]]:
        """Previous behavior: rely on Daft's to_pydict conversion."""

        result_dict = materialized.to_pydict()
        if not result_dict:
            return {}

        outputs: Dict[str, List[Any]] = {}
        for key in requested_outputs:
            column = result_dict.get(key)
            if column is None:
                continue
            outputs[key] = [self._convert_output_value(v) for v in column]
        return outputs

    def _convert_via_arrow(
        self, materialized: "daft.DataFrame", requested_outputs: List[str]
    ) -> Dict[str, List[Any]]:
        """Use PyArrow Table conversion to minimize Python overhead."""

        if not PYARROW_AVAILABLE:
            raise RuntimeError("PyArrow is required for arrow conversion")
        table = materialized.to_arrow()
        if table.num_rows == 0:
            return {}

        outputs: Dict[str, List[Any]] = {}
        for key in requested_outputs:
            if key not in table.column_names:
                continue
            column = table.column(key)
            raw_values = self._arrow_column_to_list(column)
            outputs[key] = [self._convert_output_value(v) for v in raw_values]
        return outputs

    def _convert_via_pandas(
        self, materialized: "daft.DataFrame", requested_outputs: List[str]
    ) -> Dict[str, List[Any]]:
        """Convert via pandas DataFrame for mixed object columns."""

        if not PANDAS_AVAILABLE:
            raise RuntimeError("pandas is required for pandas conversion")
        df_pd = materialized.to_pandas()
        if df_pd.empty:
            return {}

        outputs: Dict[str, List[Any]] = {}
        for key in requested_outputs:
            if key not in df_pd.columns:
                continue
            column = df_pd[key].tolist()
            outputs[key] = [self._convert_output_value(v) for v in column]
        return outputs

    def _arrow_column_to_list(self, column: "pyarrow.ChunkedArray") -> List[Any]:
        """Convert a PyArrow ChunkedArray to a Python list efficiently."""

        if not PYARROW_AVAILABLE:
            raise RuntimeError("pyarrow is required for arrow conversion")

        if pa_types and (
            pa_types.is_integer(column.type)
            or pa_types.is_floating(column.type)
            or pa_types.is_boolean(column.type)
        ):
            # Use zero-copy numpy access when possible
            return column.to_numpy(zero_copy_only=False).tolist()

        # Fall back to chunked to_pylist for complex types
        if column.num_chunks <= 1:
            return column.to_pylist()

        values: List[Any] = []
        for chunk in column.chunks:
            values.extend(chunk.to_pylist())
        return values

    def _pipeline_accepts_stateful_param(
        self, pipeline: "Pipeline", param_name: str, value: Any
    ) -> bool:
        """Check if any node in the pipeline can consume the param as stateful."""

        for node in pipeline.nodes:
            if hasattr(node, "pipeline") and hasattr(node, "input_mapping"):
                mapping = getattr(node, "input_mapping", {}) or {}
                inner_name = mapping.get(param_name)
                if inner_name and self._pipeline_accepts_stateful_param(
                    node.pipeline, inner_name, value
                ):
                    return True
            elif hasattr(node, "nodes") and hasattr(node, "run"):
                if self._pipeline_accepts_stateful_param(node, param_name, value):
                    return True
            elif hasattr(node, "func"):
                try:
                    type_hints = get_type_hints(node.func)
                except Exception:
                    type_hints = {}
                hint = type_hints.get(param_name)
                if hint and self._has_daft_cls_hint(hint):
                    return True
                if param_name in getattr(
                    node, "parameters", ()
                ) and self._has_daft_cls_hint(value):
                    return True
        return False

    def _resolve_stateful_params(
        self,
        params: List[str],
        stateful_inputs: Dict[str, Any],
        type_hints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Identify parameters that should be captured via @daft.cls."""

        resolved: Dict[str, Any] = {}
        for param in params:
            if param not in stateful_inputs:
                continue
            value = stateful_inputs[param]
            hint = type_hints.get(param)
            should_capture = self._should_capture_stateful_input(
                value
            ) or self._has_daft_cls_hint(hint)
            if should_capture:
                resolved[param] = value
            elif getattr(self, "debug", False):
                print(
                    f"[DaftEngine] Skipping stateful candidate '{param}' "
                    f"(hint={hint}, value_type={type(value)})"
                )
        if resolved and getattr(self, "debug", False):
            print(f"[DaftEngine] Captured stateful params {list(resolved.keys())}")
        return resolved

    def _has_daft_cls_hint(self, obj: Any) -> bool:
        """Return True if object/type requests @daft.cls."""

        if obj is None:
            return False
        hint = getattr(obj, "__daft_hint__", None)
        if isinstance(hint, str):
            normalized = hint.lower()
            return normalized in {"@daft.cls", "daft.cls", "cls"}
        if getattr(obj, "__daft_stateful__", False):
            return True
        cls = obj if isinstance(obj, type) else getattr(obj, "__class__", None)
        if cls is not None:
            if hasattr(cls, "__daft_cls__") or hasattr(cls, "_daft_cls"):
                return True
        return False

    def _should_capture_stateful_input(self, value: Any) -> bool:
        """Return True if the value should bypass DataFrame columns."""

        if value is None:
            return False

        if self._has_daft_cls_hint(value) or self._is_daft_cls_instance(value):
            return True

        cls = value if isinstance(value, type) else getattr(value, "__class__", None)
        if cls is not None:
            try:
                if self._class_requires_by_value_serialization(cls):
                    return True
            except Exception:
                return True

        primitive_types = (str, bytes, bool, int, float, complex)
        if isinstance(value, primitive_types) or value is None:
            return False

        if isinstance(value, (list, tuple, dict, set)):
            return False

        return True

    def _split_inputs_for_dataframe(
        self, inputs: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Set[str]]:
        """Split inputs into DataFrame columns vs. stateful-only values."""

        stateful_inputs = dict(inputs)
        df_inputs: Dict[str, Any] = {}
        stateful_only_keys: Set[str] = set()

        for key, value in inputs.items():
            if self._should_capture_stateful_input(value):
                stateful_only_keys.add(key)
            else:
                df_inputs[key] = value

        return df_inputs, stateful_inputs, stateful_only_keys

    def _extract_daft_cls_kwargs(
        self, stateful_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract @daft.cls parameters from stateful objects with sensible defaults.

        Always sets use_process=False as default for PyTorch/HuggingFace compatibility.

        Looks for optional hints on stateful objects:
        - __daft_max_concurrency__: int (limit concurrent instances)
        - __daft_use_process__: bool (override default use_process=False)
        - __daft_gpus__: int (request GPUs)

        Returns:
            Dict of kwargs to pass to @daft.cls decorator
        """
        # Start with sensible defaults
        # use_process=False is critical: avoids PyTorch/CUDA fork issues
        cls_kwargs: Dict[str, Any] = {"use_process": False}

        # Collect optional hints from stateful values
        # Fix stateful values for pickling and update dict with fixed versions
        for key, value in list(stateful_values.items()):
            fixed_value = self._ensure_stateful_value_pickleable(value)
            # Update dict if value was replaced (e.g., frozen Pydantic models)
            if fixed_value is not value:
                stateful_values[key] = fixed_value

            if fixed_value is None:
                continue

            # Check for max_concurrency hint (use fixed_value for attribute access)
            max_concurrency = getattr(fixed_value, "__daft_max_concurrency__", None)
            if max_concurrency is not None:
                # Use minimum if multiple values specify it
                if "max_concurrency" in cls_kwargs:
                    cls_kwargs["max_concurrency"] = min(
                        cls_kwargs["max_concurrency"], max_concurrency
                    )
                else:
                    cls_kwargs["max_concurrency"] = max_concurrency

            # Check for use_process hint (overrides default)
            use_process = getattr(fixed_value, "__daft_use_process__", None)
            if use_process is not None:
                # If any stateful value explicitly requires threads (use_process=False), respect it
                if use_process is False:
                    cls_kwargs["use_process"] = False
                elif "use_process" not in cls_kwargs or cls_kwargs["use_process"]:
                    # Only set to True if not already False
                    cls_kwargs["use_process"] = use_process

            # Check for GPU requirements
            gpus = getattr(fixed_value, "__daft_gpus__", None)
            if gpus is None:
                gpus = getattr(fixed_value, "gpus", None)
            if gpus is not None and isinstance(gpus, int):
                # Sum GPU requirements
                cls_kwargs["gpus"] = cls_kwargs.get("gpus", 0) + gpus

        if getattr(self, "debug", False):
            print(f"[DaftEngine] Extracted @daft.cls kwargs: {cls_kwargs}")

        return cls_kwargs

    @staticmethod
    def _class_requires_by_value_serialization(cls: type) -> bool:
        """Check if a class needs to be serialized by value (not by reference).

        Returns True if the class is from:
        - __main__ module (scripts run directly)
        - Non-importable modules
        - Modules from script files (not installed packages)

        Returns False for:
        - Built-in types
        - Standard library modules
        - Installed packages
        """
        module_name = getattr(cls, "__module__", None)
        if not module_name:
            return True
        if module_name in {"__main__", "__mp_main__"}:
            return True
        if module_name.startswith(("builtins", "typing", "_")):
            return False

        # Check if module is in sys.modules
        if module_name not in sys.modules:
            return True

        module = sys.modules[module_name]
        if not hasattr(module, "__file__"):
            # Modules injected at runtime (e.g., script modules) won't have __file__
            # and therefore won't be importable inside Daft worker processes.
            return True

        module_file = getattr(module, "__file__", None)
        if not module_file:
            return True

        # Check if the module is from an installed package
        # If it's in site-packages or dist-packages, it's installable
        if "site-packages" in module_file or "dist-packages" in module_file:
            return False

        # Check if module can be imported via standard mechanisms
        try:
            spec = importlib.util.find_spec(module_name)
            # If we can find a spec AND it's in a standard location, it's importable
            if spec and spec.origin:
                # Only return False if it's truly from an installed location
                if "site-packages" in spec.origin or "dist-packages" in spec.origin:
                    return False
        except (ImportError, ValueError, AttributeError):
            return True

        # If we got here, it's likely a script file that won't be available on workers
        return True

    def _collect_classes_from_annotation(
        self, annotation: Any, classes_to_fix: Set[type]
    ) -> None:
        """Recursively extract classes from a type annotation.

        Handles:
        - Simple types: MyClass
        - Generic types: List[MyClass], Dict[str, MyClass]
        - Union types: Union[MyClass, None], Optional[MyClass]
        - Nested generics: List[List[MyClass]]

        Args:
            annotation: Type annotation to extract classes from
            classes_to_fix: Set to add found classes to
        """
        # Simple class type
        if isinstance(annotation, type):
            module_name = getattr(annotation, "__module__", "unknown")
            if self._class_requires_by_value_serialization(annotation):
                if self.debug:
                    print(
                        "[_collect_classes_from_annotation] Found class to fix: "
                        f"{module_name}.{annotation.__name__}"
                    )
                classes_to_fix.add(annotation)
            else:
                if self.debug:
                    print(
                        "[_collect_classes_from_annotation] Skipping "
                        f"{module_name}.{annotation.__name__} (does not require fixing)"
                    )
            return

        # Check for generic types (List, Dict, Optional, etc.)
        origin = get_origin(annotation)
        if origin is not None:
            args = get_args(annotation)
            if args:
                # Recursively process all type arguments
                for arg in args:
                    self._collect_classes_from_annotation(arg, classes_to_fix)

    def _fix_all_script_classes(
        self, pipeline: "Pipeline", inputs: Dict[str, Any]
    ) -> None:
        """Proactively fix all classes from unimportable script modules.

        This scans through all inputs and their types, finds classes with
        unimportable module names, and fixes them in place by:
        1. Changing their __module__ to "__main__"
        2. Rebinding them in sys.modules

        This MUST happen before Daft sees any of these classes, otherwise
        Daft will embed the original module names in its expression tree,
        causing deserialization failures on workers.

        Args:
            pipeline: The pipeline to scan for type annotations
            inputs: Dictionary of input values to scan
        """
        import sys
        import types

        classes_to_fix: Set[type] = set()
        visited: Set[int] = set()

        def collect_classes(obj: Any, depth: int = 0) -> None:
            """Recursively collect all classes that need fixing."""
            if depth > 10:  # Prevent infinite recursion
                return

            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)

            # Get the class of the object
            if isinstance(obj, type):
                # obj itself is a class
                cls = obj
            else:
                # obj is an instance
                cls = type(obj)

            # Check if this class needs fixing
            if self._class_requires_by_value_serialization(cls):
                classes_to_fix.add(cls)

            # Recursively scan object attributes
            try:
                if hasattr(obj, "__dict__"):
                    for attr_value in obj.__dict__.values():
                        collect_classes(attr_value, depth + 1)

                # Scan list/tuple/set items
                if isinstance(obj, (list, tuple, set)):
                    for item in obj:
                        collect_classes(item, depth + 1)

                # Scan dict values
                if isinstance(obj, dict):
                    for value in obj.values():
                        collect_classes(value, depth + 1)

                # Scan Pydantic model fields
                if PYDANTIC_AVAILABLE and isinstance(obj, BaseModel):
                    for field_value in obj.__dict__.values():
                        collect_classes(field_value, depth + 1)
            except (AttributeError, TypeError):
                pass

        # Collect all classes from inputs
        for value in inputs.values():
            collect_classes(value)

        # CRITICAL: Also scan pipeline node function annotations
        # Pydantic models and other types used in signatures won't be in inputs
        if self.debug:
            print(
                f"[_fix_all_script_classes] Scanning {len(pipeline.nodes)} nodes for type annotations"
            )

        for node_obj in pipeline.nodes:
            try:
                # Get the original function
                if hasattr(node_obj, "func"):
                    func = node_obj.func
                    # Get type hints from the function
                    if hasattr(func, "__annotations__"):
                        if self.debug and func.__annotations__:
                            print(
                                f"[_fix_all_script_classes] Node {node_obj.output_name} has annotations: {list(func.__annotations__.keys())}"
                            )

                        # Try to get evaluated type hints (handles string annotations)
                        try:
                            type_hints = get_type_hints(func)
                            for annotation in type_hints.values():
                                self._collect_classes_from_annotation(
                                    annotation, classes_to_fix
                                )
                        except Exception as e:
                            if self.debug:
                                print(
                                    f"[_fix_all_script_classes] Could not get type hints for {node_obj.output_name}: {e}"
                                )
                            # Fall back to raw annotations
                            for annotation in func.__annotations__.values():
                                self._collect_classes_from_annotation(
                                    annotation, classes_to_fix
                                )
            except Exception as e:
                if self.debug:
                    print(f"[_fix_all_script_classes] Error scanning node: {e}")

        # Fix each class in place
        if self.debug:
            print(
                f"[_fix_all_script_classes] Found {len(classes_to_fix)} classes to potentially fix"
            )

        for cls in classes_to_fix:
            original_module = cls.__module__

            if original_module in ("__main__", "__mp_main__"):
                if self.debug:
                    print(
                        f"[_fix_all_script_classes] Skipping {cls.__name__} (already {original_module})"
                    )
                continue

            try:
                # Get or create the original module object
                if original_module in sys.modules:
                    original_module_obj = sys.modules[original_module]
                else:
                    # Module doesn't exist, create a placeholder
                    original_module_obj = types.ModuleType(original_module)
                    sys.modules[original_module] = original_module_obj

                # Ensure __main__ exists
                if "__main__" not in sys.modules:
                    sys.modules["__main__"] = types.ModuleType("__main__")
                main_module = sys.modules["__main__"]

                # Change the class's __module__ to __main__
                cls.__module__ = "__main__"

                # Rebind in both modules to maintain references
                class_name = cls.__name__
                setattr(main_module, class_name, cls)
                setattr(original_module_obj, class_name, cls)

                if self.debug:
                    print(
                        f"[_fix_all_script_classes] Fixed {original_module}.{class_name} -> __main__.{class_name}"
                    )
                    # Verify the fix
                    if cls.__module__ != "__main__":
                        print(
                            f"[_fix_all_script_classes] WARNING: {class_name}.__module__ is still {cls.__module__}, not __main__!"
                        )

            except Exception as e:
                if self.debug:
                    print(
                        f"[_fix_all_script_classes] Warning: Failed to fix {cls.__name__}: {e}"
                    )

    def _register_all_script_modules(self, inputs: Dict[str, Any]) -> None:
        """Proactively register all script modules with cloudpickle.

        This method walks through all input values, finds custom class instances,
        and registers their modules with cloudpickle BEFORE Daft starts creating
        Expression objects. This prevents ModuleNotFoundError in UDF workers.

        Args:
            inputs: Dictionary of pipeline inputs
        """
        try:
            import sys

            import cloudpickle

            modules_to_register = set()

            def collect_modules(obj: Any, visited: Optional[Set[int]] = None) -> None:
                """Recursively collect all module names from custom objects."""
                if visited is None:
                    visited = set()

                obj_id = id(obj)
                if obj_id in visited:
                    return
                visited.add(obj_id)

                # Skip builtin types
                if _is_builtin_type(obj):
                    return

                # Get module from class
                try:
                    cls = type(obj)
                    module_name = cls.__module__

                    # Check if module needs registration (not builtin/importable)
                    if module_name not in (
                        "__main__",
                        "__mp_main__",
                        "builtins",
                        "__builtin__",
                        "typing",
                    ):
                        try:
                            spec = importlib.util.find_spec(module_name)
                            if spec is None:
                                # Module not importable - needs registration
                                modules_to_register.add(module_name)
                        except (ImportError, ValueError, AttributeError):
                            # Module not importable - needs registration
                            modules_to_register.add(module_name)

                    # Recursively check attributes
                    if hasattr(obj, "__dict__"):
                        for attr_value in obj.__dict__.values():
                            if not _is_builtin_type(attr_value):
                                if isinstance(attr_value, (list, tuple)):
                                    for item in attr_value:
                                        collect_modules(item, visited)
                                elif isinstance(attr_value, dict):
                                    for v in attr_value.values():
                                        collect_modules(v, visited)
                                else:
                                    collect_modules(attr_value, visited)
                except (AttributeError, TypeError):
                    pass

            # Collect all modules from inputs
            for value in inputs.values():
                collect_modules(value)

            # Register all found modules with cloudpickle
            for module_name in modules_to_register:
                if module_name in sys.modules:
                    try:
                        module = sys.modules[module_name]
                        cloudpickle.register_pickle_by_value(module)
                        if getattr(self, "debug", False):
                            print(
                                f"[DaftEngine] Registered module '{module_name}' for by-value pickling"
                            )
                    except Exception as e:
                        if getattr(self, "debug", False):
                            print(
                                f"[DaftEngine] Warning: Failed to register module '{module_name}': {e}"
                            )

        except ImportError:
            # cloudpickle not available
            pass
        except Exception as e:
            if getattr(self, "debug", False):
                print(f"[DaftEngine] Warning: Module registration failed: {e}")

    def _ensure_stateful_value_pickleable(self, value: Any) -> Any:
        """Fix stateful value instances comprehensively for Daft serialization.

        This uses the new comprehensive approach that:
        1. Recursively fixes all nested custom objects
        2. Registers modules with cloudpickle
        3. Updates sys.modules for class lookup

        Returns:
            The fixed value ready for Daft serialization
        """
        return _prepare_stateful_value_for_daft(
            value, debug=getattr(self, "debug", False)
        )

    def _build_stateful_udf(
        self,
        func: Any,
        params: List[str],
        stateful_values: Dict[str, Any],
        inferred_dtype: Optional["daft.DataType"],
    ):
        """Create a @daft.cls wrapper that captures stateful parameters.

        Automatically extracts concurrency hints from stateful objects and applies
        them to the @daft.cls decorator. Defaults to use_process=False for
        PyTorch/HuggingFace compatibility.
        """

        import daft

        # Force by-value serialization for Modal/distributed execution
        serializable_func = _make_serializable_by_value(func)

        # Comprehensively prepare all stateful values for serialization
        # This includes deep recursive fixing of nested objects
        if self.debug:
            print(
                f"[_build_stateful_udf] Preparing {len(stateful_values)} stateful values:"
            )
            for key, value in stateful_values.items():
                cls = value.__class__
                print(f"  - {key}: {cls.__module__}.{cls.__name__}")

        prepared_stateful_values = {
            key: self._ensure_stateful_value_pickleable(value)
            for key, value in stateful_values.items()
        }

        if self.debug:
            print("[_build_stateful_udf] After preparation:")
            for key, value in prepared_stateful_values.items():
                cls = value.__class__
                print(f"  - {key}: {cls.__module__}.{cls.__name__}")

        # Extract @daft.cls parameters from stateful objects (with sensible defaults)
        cls_kwargs = self._extract_daft_cls_kwargs(prepared_stateful_values)

        method_kwargs: Dict[str, Any] = {}
        if inferred_dtype is None:
            inferred_dtype = daft.DataType.python()
        method_kwargs["return_dtype"] = inferred_dtype

        method_decorator = (
            daft.method(**method_kwargs) if method_kwargs else daft.method()
        )

        # Capture debug flag for logging inside the wrapper
        debug_mode = self.debug

        # Calculate dynamic params (those not in stateful_values)
        dynamic_params = [p for p in params if p not in prepared_stateful_values]

        @daft.cls(**cls_kwargs)
        class StatefulWrapper:
            def __init__(self, payload: Dict[str, Any], dynamic_param_names: List[str]):
                self._payload = payload
                self._dynamic_params = dynamic_param_names

            @method_decorator
            def __call__(self, *args):
                # Build kwargs by combining stateful values with dynamic args
                # The args correspond to dynamic_params in order
                kwargs = dict(self._payload)  # Start with stateful values

                # Add dynamic params from args
                if len(args) != len(self._dynamic_params):
                    raise ValueError(
                        f"Expected {len(self._dynamic_params)} dynamic args "
                        f"({self._dynamic_params}), got {len(args)}"
                    )
                for param_name, arg_value in zip(self._dynamic_params, args):
                    kwargs[param_name] = arg_value

                # Call the function
                result = serializable_func(**kwargs)

                # CRITICAL FIX: Apply _fix_output_tree to ensure outputs are serializable
                # This fixes script-defined classes (Pydantic models, etc.) that are
                # returned from the function
                if debug_mode:
                    print("[StatefulWrapper] Fixing output from stateful UDF")
                fixed_result = _fix_output_tree(result, debug=debug_mode)

                return fixed_result

        wrapper = StatefulWrapper(prepared_stateful_values, dynamic_params)
        return wrapper

    def _should_execute_stateful_in_python(
        self, stateful_values: Dict[str, Any]
    ) -> bool:
        """Return True if the node should run in Python instead of Daft for stability."""

        for value in stateful_values.values():
            try:
                cls = value if isinstance(value, type) else value.__class__
                if getattr(cls, "__hn_from_script__", False):
                    if getattr(self, "debug", False):
                        print(
                            f"[DaftEngine] Executing stateful node for script class: {cls.__name__}"
                        )
                    return True
                if getattr(value, "__hn_fixed_by_value__", False):
                    if getattr(self, "debug", False):
                        print(
                            f"[DaftEngine] Executing stateful node for fixed value: {cls.__name__}"
                        )
                    return True
                if self._class_requires_by_value_serialization(cls):
                    if getattr(self, "debug", False):
                        print(
                            f"[DaftEngine] Executing stateful node due to by-value serialization: {cls.__name__}"
                        )
                    return True
            except Exception:
                return True
        return False

    def _execute_stateful_node_in_python(
        self,
        df: "daft.DataFrame",
        dynamic_params: List[str],
        stateful_values: Dict[str, Any],
        func: Any,
        output_name: str,
        inferred_dtype: Optional["daft.DataType"],
    ) -> "daft.DataFrame":
        """Execute a stateful node row-by-row in Python and materialize the results."""

        # Handle no-column case (pure stateful)
        if not dynamic_params:
            result = func(**stateful_values)
            result = _fix_output_tree(result, debug=self.debug)
            import daft

            expr = daft.lit(result)
            if inferred_dtype is not None:
                expr = expr.cast(inferred_dtype)
            return df.with_column(output_name, expr)

        # Fetch required columns as Python data
        debug_mode = getattr(self, "debug", False)
        if debug_mode:
            print(
                f"[DaftEngine] Fetching columns {dynamic_params} for Python execution of '{output_name}'"
            )
        subset = df.select(*dynamic_params).to_pydict()
        if not subset:
            return df

        num_rows = len(next(iter(subset.values())))
        results: List[Any] = []
        for idx in range(num_rows):
            kwargs = {param: subset[param][idx] for param in dynamic_params}
            kwargs.update(stateful_values)
            value = func(**kwargs)
            value = _fix_output_tree(value, debug=self.debug)
            results.append(value)
        if debug_mode:
            print(
                f"[DaftEngine] Executed '{output_name}' in Python for {num_rows} rows"
            )

        # Materialize existing columns so we can rebuild a DataFrame that already
        # contains the Python results.
        data = df.to_pydict()
        if not data:
            data = {param: subset[param] for param in dynamic_params}
        data[output_name] = results

        import daft

        return daft.from_pydict(data)

    def _infer_return_dtype(self, return_type: Any) -> Optional["daft.DataType"]:
        """Infer an appropriate Daft DataType for a node return annotation.

        Now with native Pydantic model support!
        """

        import daft

        origin = get_origin(return_type)

        # Handle Optional[...] (i.e., Union[..., None])
        if origin is Union:
            args = [arg for arg in get_args(return_type) if arg is not type(None)]  # noqa: E721
            if len(args) == 1:
                return self._infer_return_dtype(args[0])
            # Multiple options – fall back to Python storage
            return daft.DataType.python()

        if origin in (list, List):
            elem_type = get_args(return_type)[0] if get_args(return_type) else Any
            elem_dtype = self._infer_return_dtype(elem_type)
            if elem_dtype is None:
                elem_dtype = daft.DataType.python()
            return daft.DataType.list(elem_dtype)

        if origin in (dict, Dict, tuple, Tuple):
            return daft.DataType.python()

        # Handle builtin generics without typing annotations
        if return_type is dict:
            return daft.DataType.python()

        if return_type is list:
            return daft.DataType.list(daft.DataType.python())

        if return_type is tuple:
            return daft.DataType.python()

        if return_type in (int, bool):
            return daft.DataType.int64() if return_type is int else daft.DataType.bool()

        if return_type is float:
            return daft.DataType.float64()

        if return_type is str:
            return daft.DataType.string()

        if return_type is Any:
            return daft.DataType.python()

        # NEW: Handle Pydantic models
        if (
            PYDANTIC_AVAILABLE
            and BaseModel is not None
            and isinstance(return_type, type)
        ):
            try:
                if issubclass(return_type, BaseModel):
                    # Check if the model has arbitrary_types_allowed
                    # If so, use Python object storage instead of PyArrow structs
                    has_arbitrary_types = False
                    if hasattr(return_type, "model_config"):
                        config = return_type.model_config
                        if isinstance(config, dict):
                            has_arbitrary_types = config.get(
                                "arbitrary_types_allowed", False
                            )

                    # IMPORTANT: Check if model is from a non-importable module (like a script)
                    # In that case, ALWAYS use Python storage to avoid storing class references
                    # in Daft's execution plan that can't be unpickled in workers
                    is_from_script = False
                    try:
                        module_name = return_type.__module__
                        if module_name and module_name not in (
                            "__main__",
                            "__mp_main__",
                            "builtins",
                        ):
                            spec = importlib.util.find_spec(module_name)
                            if spec is None:
                                is_from_script = True
                    except (ImportError, ValueError, AttributeError):
                        is_from_script = True

                    # Use Python object storage for models with arbitrary types OR from scripts
                    # (like numpy arrays or non-importable modules) to avoid serialization issues
                    if has_arbitrary_types or is_from_script:
                        return daft.DataType.python()

                    # Try PyArrow struct conversion for simple Pydantic models
                    if (
                        PYARROW_AVAILABLE
                        and PYDANTIC_TO_PYARROW_AVAILABLE
                        and get_pyarrow_schema is not None
                    ):
                        try:
                            schema = get_pyarrow_schema(return_type)
                            arrow_type = pyarrow.struct(
                                [(f, schema.field(f).type) for f in schema.names]
                            )
                            return daft.DataType.from_arrow_type(arrow_type)
                        except (TypeError, Exception):
                            # If PyArrow conversion fails, fall back to Python object storage
                            return daft.DataType.python()
                    else:
                        # If pydantic_to_pyarrow not available, use Python object storage
                        return daft.DataType.python()
            except (TypeError, Exception):
                # If conversion fails or not a Pydantic model, continue
                pass

        if isinstance(return_type, type) and return_type.__module__ != "builtins":
            return daft.DataType.python()

        # Default: let Daft infer
        return None

    # ==================== Code Generation Methods ====================

    def _init_code_generation(self):
        """Initialize code generation tracking."""
        self.generated_code: List[str] = []
        self._udf_definitions: List[str] = []
        self._imports: Set[Tuple[str, str]] = set()  # (module, name) tuples
        self._stateful_objects: Dict[str, str] = {}  # name -> init code
        self._udf_counter = 0
        self._row_id_counter = 0
        self._actual_inputs: Optional[Dict[str, Any]] = None
        self._actual_output_name: Union[str, List[str], None] = None
        self._stateful_input_names: Set[str] = set()  # Track which inputs are stateful
        self._stateful_actual_inputs: Dict[str, Any] = {}
        self._stateful_literal_input_keys: Set[str] = set()

    def _add_import(self, module: str, names: Union[str, List[str]]):
        """Track imports needed for generated code."""
        if not self.code_generation_mode:
            return

        if isinstance(names, str):
            names = [names]

        for name in names:
            self._imports.add((module, name))

    def _generate_udf_name(self, base_name: str) -> str:
        """Generate unique UDF name."""
        self._udf_counter += 1
        return f"{base_name}_{self._udf_counter}"

    def _generate_row_id_name(self) -> str:
        """Generate unique row ID column name."""
        self._row_id_counter += 1
        return f"__daft_row_id_{self._row_id_counter}__"

    def _format_dtype_for_code(self, dtype: Optional["daft.DataType"]) -> str:
        """Return a valid Python expression for a Daft DataType."""
        import daft

        if dtype is None:
            return "daft.DataType.python()"

        simple_map = {
            daft.DataType.python(): "daft.DataType.python()",
            daft.DataType.bool(): "daft.DataType.bool()",
            daft.DataType.int8(): "daft.DataType.int8()",
            daft.DataType.int16(): "daft.DataType.int16()",
            daft.DataType.int32(): "daft.DataType.int32()",
            daft.DataType.int64(): "daft.DataType.int64()",
            daft.DataType.uint8(): "daft.DataType.uint8()",
            daft.DataType.uint16(): "daft.DataType.uint16()",
            daft.DataType.uint32(): "daft.DataType.uint32()",
            daft.DataType.uint64(): "daft.DataType.uint64()",
            daft.DataType.float32(): "daft.DataType.float32()",
            daft.DataType.float64(): "daft.DataType.float64()",
            daft.DataType.string(): "daft.DataType.string()",
        }

        for sample, literal in simple_map.items():
            if dtype == sample:
                return literal

        if dtype.is_list():
            inner = self._format_dtype_for_code(dtype.dtype)
            return f"daft.DataType.list({inner})"

        if dtype.is_struct():
            fields = [
                f'"{name}": {self._format_dtype_for_code(field_dtype)}'
                for name, field_dtype in dtype.fields.items()
            ]
            joined = ", ".join(fields)
            return f"daft.DataType.struct({{{joined}}})"

        if dtype.is_map():
            key_code = self._format_dtype_for_code(dtype.key_type)
            value_code = self._format_dtype_for_code(dtype.value_type)
            return f"daft.DataType.map({key_code}, {value_code})"

        return "daft.DataType.python()"

    def _extract_function_body(self, func: Any) -> Optional[str]:
        """Return dedented source for the body of ``func``."""
        import inspect

        try:
            raw_source = inspect.getsource(func)
        except (OSError, TypeError):
            return None

        dedented = textwrap.dedent(raw_source)
        try:
            module = ast.parse(dedented)
        except SyntaxError:
            return None

        func_nodes = [
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if not func_nodes:
            return None

        func_node = func_nodes[0]
        if not func_node.body:
            return None

        lines = dedented.split("\n")
        start = max(func_node.body[0].lineno - 1, 0)
        end_node = func_node.body[-1]
        end = getattr(end_node, "end_lineno", end_node.lineno)
        end = min(end, len(lines))
        body = "\n".join(lines[start:end])
        return body

    def _generate_udf_code(
        self,
        func: Any,
        output_name: str,
        params: List[str],
        stateful_values: Dict[str, Any],
        inferred_dtype: Optional["daft.DataType"],
    ) -> str:
        """Generate UDF definition code with proper decorators.

        Returns:
            UDF name to use in with_column calls
        """
        if self.code_generation_mode:
            self._add_import("typing", "Any")

        func_name = func.__name__
        udf_name = self._generate_udf_name(func_name)

        # Generate code based on whether it's stateful or not
        if stateful_values:
            # Generate @daft.cls wrapper
            class_name = f"{func_name.title().replace('_', '')}Wrapper"

            lines = []
            lines.append("@daft.cls(use_process=False)")
            lines.append(f"class {class_name}:")

            # __init__ with stateful parameters
            stateful_params = ", ".join([f"{k}: Any" for k in stateful_values.keys()])
            lines.append(f"    def __init__(self, {stateful_params}):")
            for k in stateful_values.keys():
                lines.append(f"        self.{k} = {k}")
            lines.append("")

            # __call__ method (or named method)
            dtype_str = self._format_dtype_for_code(inferred_dtype)
            dynamic_params = [p for p in params if p not in stateful_values]
            method_params = ", ".join([f"{p}: Any" for p in dynamic_params])

            lines.append(f"    @daft.method(return_dtype={dtype_str})")
            lines.append(f"    def __call__(self, {method_params}):")

            # Generate function call
            all_params = ", ".join(
                [f"self.{p}" if p in stateful_values else p for p in params]
            )
            lines.append(f"        return {func_name}({all_params})")
            lines.append("")

            # Instantiate wrapper and expose bound __call__ just like runtime execution
            init_args = ", ".join([f"{k}={k}" for k in stateful_values.keys()])
            wrapper_instance = f"{udf_name}_wrapper"
            lines.append(f"{wrapper_instance} = {class_name}({init_args})")
            lines.append(f"{udf_name} = {wrapper_instance}")

            # Emit a small helper function so the generated code shows an explicit signature.
            helper_name = f"{udf_name}_signature"
            if method_params:
                lines.append("")
                lines.append(f"def {helper_name}({method_params}):")
            else:
                lines.append("")
                lines.append(f"def {helper_name}():")
            lines.append(
                '    """Helper to illustrate how to invoke this stateful UDF."""'
            )
            call_args = ", ".join(dynamic_params)
            if call_args:
                lines.append(f"    return {udf_name}({call_args})")
            else:
                lines.append(f"    return {udf_name}()")

            self._udf_definitions.append("\n".join(lines))
            return udf_name
        else:
            # Generate @daft.func
            lines = []

            # Format parameters
            param_list = ", ".join([f"{p}: Any" for p in params])

            dtype_str = self._format_dtype_for_code(inferred_dtype)
            lines.append(f"@daft.func(return_dtype={dtype_str})")
            lines.append(f"def {udf_name}({param_list}):")

            # Try to get function body
            body = self._extract_function_body(func)
            if body:
                body = textwrap.dedent(body)
                for line in body.split("\n"):
                    lines.append(f"    {line}" if line else "")
            else:
                lines.append("    # Source code not available")
                lines.append(f"    return {func_name}({', '.join(params)})")

            self._udf_definitions.append("\n".join(lines))
            return udf_name

    def _generate_operation_code(self, operation: str):
        """Add a DataFrame operation line to generated code."""
        if not self.code_generation_mode:
            return

        self.generated_code.append(operation)

    def _generate_dataframe_creation_lines(
        self, inputs: Dict[str, Any], lines: List[str]
    ):
        """Generate the initial DataFrame creation code with actual input data."""
        if not self.code_generation_mode:
            return

        stateful_literal_keys = getattr(self, "_stateful_literal_input_keys", set())

        # Separate inputs by type:
        # - simple: primitives that can be serialized directly
        # - daft_cls: @daft.cls instances (stateful UDFs)
        # - complex: other objects that need pre-initialization
        simple_inputs = {}
        daft_cls_inputs = {}
        complex_inputs = {}

        for key, value in inputs.items():
            if key in stateful_literal_keys:
                continue
            # Check if it's a simple type that can be serialized
            if self._is_simple_type(value):
                simple_inputs[key] = value
            elif self._is_daft_cls_instance(value):
                daft_cls_inputs[key] = value
            else:
                complex_inputs[key] = value

        # Generate the from_pydict call with simple data
        lines.append("# Create DataFrame with input data")
        lines.append("df = daft.from_pydict({")

        for key, value in simple_inputs.items():
            # Format value as Python literal
            value_repr = self._format_value_for_code(value)
            lines.append(f'    "{key}": [{value_repr}],')

        lines.append("})")
        lines.append("")

        if stateful_literal_keys:
            lines.append("# Stateful inputs handled outside the DataFrame:")
            for key in stateful_literal_keys:
                obj = self._stateful_actual_inputs.get(key)
                type_name = type(obj).__name__ if obj is not None else "object"
                lines.append(f"#   - {key}: {type_name} (captured via @daft.cls)")
            lines.append("")

        # Add warnings for @daft.cls instances
        if daft_cls_inputs:
            lines.append(
                "# ==================== ⚠️  PERFORMANCE WARNING ===================="
            )
            lines.append(
                "# The following stateful objects are being passed via daft.lit():"
            )
            for key in daft_cls_inputs.keys():
                lines.append(f"#   - {key}: {type(daft_cls_inputs[key]).__name__}")
            lines.append("#")
            lines.append(
                "# This is INEFFICIENT! The object is serialized into every row."
            )
            lines.append(
                "# These objects are already @daft.cls instances and should be"
            )
            lines.append("# used DIRECTLY without daft.lit().")
            lines.append("#")
            lines.append("# CORRECT USAGE:")
            lines.append(
                f"#   {list(daft_cls_inputs.keys())[0]} = {type(list(daft_cls_inputs.values())[0]).__name__}(...)"
            )
            lines.append(
                f"#   df = df.with_column('result', {list(daft_cls_inputs.keys())[0]}(df['input']))"
            )
            lines.append("#")
            lines.append("# Do NOT add them as columns with daft.lit()!")
            lines.append(
                "# ================================================================"
            )
            lines.append("")

        # Add complex inputs as DataFrame columns (they need to be pre-initialized)
        if complex_inputs or daft_cls_inputs:
            if complex_inputs:
                lines.append("# Add complex/stateful objects as columns")
                lines.append(
                    "# Note: These objects must be initialized before running this code"
                )
                for key in complex_inputs.keys():
                    lines.append(f'df = df.with_column("{key}", daft.lit({key}))')
                lines.append("")

            # Include daft_cls for now (with warning above)
            if daft_cls_inputs:
                lines.append(
                    "# ⚠️  WARNING: @daft.cls objects below - see warning above!"
                )
                for key in daft_cls_inputs.keys():
                    lines.append(
                        f'df = df.with_column("{key}", daft.lit({key}))  # ← INEFFICIENT!'
                    )
                lines.append("")

    def _is_simple_type(self, value: Any) -> bool:
        """Check if a value is a simple serializable type."""
        return isinstance(value, (int, float, str, bool, type(None), list, tuple, dict))

    def _is_daft_cls_instance(self, value: Any) -> bool:
        """Check if value is a @daft.cls instance."""
        # Check for @daft.cls marker
        if hasattr(value, "__class__"):
            cls = value.__class__
            # Daft marks classes with special attributes
            return hasattr(cls, "__daft_cls__") or hasattr(cls, "_daft_cls")
        return False

    def _generate_output_collection(self, requested_outputs: List[str]):
        """Generate the output collection code."""
        if not self.code_generation_mode:
            return

        self.generated_code.append("")
        self.generated_code.append("# Select output columns")

        if len(requested_outputs) == 1:
            self.generated_code.append(f'df = df.select(df["{requested_outputs[0]}"])')
        else:
            cols = ", ".join([f'df["{out}"]' for out in requested_outputs])
            self.generated_code.append(f"df = df.select({cols})")

        self.generated_code.append("")
        self.generated_code.append("# Collect results")
        self.generated_code.append("result = df.collect()")
        self.generated_code.append("print(result.to_pydict())")

    def _format_value_for_code(self, value: Any) -> str:
        """Format a value as Python code string.

        Returns a valid Python literal that can be used in generated code.
        Unlike reprlib.repr(), this returns the FULL representation without
        truncation, since we need executable code not debug output.
        """
        # Handle common types
        if isinstance(value, str):
            return repr(value)
        elif isinstance(value, (int, float, bool)):
            return repr(value)
        elif isinstance(value, (list, tuple)):
            # Use full repr for code generation (not reprlib which truncates with ...)
            return repr(value)
        elif value is None:
            return "None"
        else:
            # For complex objects, just use a placeholder
            return f"<{type(value).__name__} object>"

    def get_generated_code(self) -> str:
        """Return complete executable Daft code that exactly matches the pipeline execution."""
        if not self.code_generation_mode:
            return "# Code generation mode not enabled"

        lines = []

        # Header with performance analysis
        lines.append('"""')
        lines.append(
            "Generated Daft code - Exact translation from HyperNodes pipeline."
        )
        lines.append("")
        lines.append(
            "This code produces identical results to the HyperNodes pipeline execution."
        )
        lines.append("You can run this file directly to verify the translation.")
        lines.append("")
        lines.append("=" * 70)
        lines.append("PERFORMANCE ANALYSIS")
        lines.append("=" * 70)

        # Count map operations
        map_count = sum(1 for line in self.generated_code if "# Map over:" in line)
        if map_count > 0:
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
        lines.append("")

        # Imports
        lines.append("import daft")

        # Group imports by module
        imports_by_module: Dict[str, List[str]] = {}
        for module, name in sorted(self._imports):
            if module not in imports_by_module:
                imports_by_module[module] = []
            imports_by_module[module].append(name)

        for module, names in sorted(imports_by_module.items()):
            lines.append(f"from {module} import {', '.join(sorted(set(names)))}")

        lines.append("")

        # Stateful object setup
        if self._stateful_input_names and self._actual_inputs:
            lines.append("# ==================== Stateful Objects ====================")
            lines.append(
                "# These objects need to be initialized before running the pipeline"
            )
            lines.append("")
            for name in sorted(self._stateful_input_names):
                if name in self._actual_inputs:
                    obj = self._actual_inputs[name]
                    lines.append(
                        "# " + name + " = <" + type(obj).__name__ + " instance>"
                    )
                    lines.append(
                        "# You need to initialize this with the same configuration"
                    )
            lines.append("")

        # UDF definitions
        if self._udf_definitions:
            lines.append("# ==================== UDF Definitions ====================")
            lines.append("")
            for udf_def in self._udf_definitions:
                lines.append(udf_def)
                lines.append("")

        # Main execution
        lines.append("")
        lines.append("# ==================== Pipeline Execution ====================")
        lines.append("")

        # Pipeline operations (includes DataFrame creation and output collection)
        if self.generated_code:
            for code_line in self.generated_code:
                lines.append(code_line)

        return "\n".join(lines)
