"""Quick verification that the Modal deserialization fix works."""

import io
import pickle

import cloudpickle


def main():
    print("Verifying Modal Backend Deserialization Fix")
    print("=" * 60)
    
    # Test 1: Verify pickle.Unpickler exists and works
    print("\n1. Testing pickle.Unpickler...")
    
    class TestUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            return super().find_class(module, name)
    
    data = {"result": 42, "status": "success"}
    serialized = cloudpickle.dumps(data)
    result = TestUnpickler(io.BytesIO(serialized)).load()
    
    assert result == data
    print("   ✓ pickle.Unpickler works correctly")
    
    # Test 2: Verify ModalBackend can be imported and has the method
    print("\n2. Testing ModalBackend import...")
    from hypernodes.backend import ModalBackend
    
    assert hasattr(ModalBackend, '_deserialize_results')
    print("   ✓ ModalBackend._deserialize_results exists")
    
    # Test 3: Verify the method works with mock data
    print("\n3. Testing deserialization with ModalBackend...")
    import modal
    
    image = modal.Image.debian_slim(python_version="3.12")
    backend = ModalBackend(image=image)
    
    test_data = {"doubled": 10, "result": 20}
    serialized = cloudpickle.dumps(test_data)
    result = backend._deserialize_results(serialized)
    
    assert result == test_data
    print("   ✓ Deserialization works correctly")
    print(f"   Result: {result}")
    
    print("\n" + "=" * 60)
    print("✓ All verifications passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
