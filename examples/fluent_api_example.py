"""Example demonstrating fluent/builder API for Pipeline configuration."""

from hypernodes import Pipeline, node, DiskCache, LocalBackend
from pathlib import Path

# Define simple nodes
@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

@node(output_name="result")
def add_ten(doubled: int) -> int:
    return doubled + 10


def example_basic_fluent():
    """Basic fluent API usage."""
    print("=" * 60)
    print("Example 1: Basic Fluent API")
    print("=" * 60)
    
    # Fluent way - more readable for complex configurations
    pipeline_fluent = (
        Pipeline(nodes=[double, add_ten])
        .with_backend(LocalBackend())
        .with_cache(DiskCache(path=".cache"))
    )
    
    result = pipeline_fluent.run(inputs={"x": 5})
    print(f"Result: {result}")
    print("✓ Fluent API works!\n")


def example_chaining():
    """Method chaining for complex configurations."""
    print("=" * 60)
    print("Example 2: Method Chaining")
    print("=" * 60)
    
    # Create pipeline with chained configuration
    pipeline = (
        Pipeline(nodes=[double, add_ten])
        .with_backend(LocalBackend())
        .with_cache(DiskCache(path=".cache"))
    )
    
    # Run first time (cache miss)
    print("First run (cache miss):")
    result1 = pipeline.run(inputs={"x": 10})
    print(f"  Result: {result1}")
    
    # Run second time (cache hit)
    print("Second run (cache hit):")
    result2 = pipeline.run(inputs={"x": 10})
    print(f"  Result: {result2}")
    
    assert result1 == result2
    print("✓ Caching works with fluent API!\n")


def example_modal_backend():
    """Example with Modal backend configuration."""
    print("=" * 60)
    print("Example 3: Modal Backend (Conceptual)")
    print("=" * 60)
    
    # This shows the syntax (won't run without modal installed)
    # Commenting out to avoid import errors
    
    print("""
    Example usage with Modal:
    
    import modal
    from hypernodes import Pipeline, ModalBackend, node, DiskCache
    
    image = modal.Image.debian_slim().pip_install("numpy")
    
    pipeline = (
        Pipeline(nodes=[process, transform, aggregate])
        .with_backend(ModalBackend(
            image=image,
            gpu="A100",
            memory="32GB"
        ))
        .with_cache(DiskCache(path="./remote-cache"))
    )
    
    result = pipeline.run(inputs={...})
    """)
    print("✓ Fluent API makes complex configurations readable!\n")


def example_conditional_config():
    """Conditionally configure based on environment."""
    print("=" * 60)
    print("Example 4: Conditional Configuration")
    print("=" * 60)
    
    import os
    use_cache = os.getenv("USE_CACHE", "true").lower() == "true"
    
    # Start with basic pipeline
    pipeline = Pipeline(nodes=[double, add_ten])
    
    # Conditionally add cache
    if use_cache:
        pipeline = pipeline.with_cache(DiskCache(path=".cache"))
        print("✓ Cache enabled")
    else:
        print("✓ Cache disabled")
    
    result = pipeline.run(inputs={"x": 7})
    print(f"Result: {result}\n")


if __name__ == "__main__":
    # Clean up cache from previous runs
    import shutil
    cache_dir = Path(".cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    
    example_basic_fluent()
    example_chaining()
    example_modal_backend()
    example_conditional_config()
    
    # Clean up
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    
    print("=" * 60)
    print("All fluent API examples completed!")
    print("=" * 60)
