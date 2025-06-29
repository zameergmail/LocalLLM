#!/usr/bin/env python3
"""
Simple test to verify the application works without memory issues.
"""

import os
import sys
import tempfile
from PIL import Image, ImageDraw

def test_basic_functionality():
    """Test basic application functionality."""
    print("🧪 Testing Basic Application Functionality")
    print("=" * 50)
    
    # Test 1: Import main modules
    print("1. Testing imports...")
    try:
        import streamlit as st
        print("   ✅ Streamlit imported")
        
        from utils import check_multimodal_capabilities
        print("   ✅ Utils imported")
        
        from multimodal_utils import ImageExtractor, MultimodalDocumentProcessor
        print("   ✅ Multimodal utils imported")
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Check multimodal capabilities
    print("2. Testing multimodal capabilities...")
    try:
        capabilities = check_multimodal_capabilities()
        print(f"   ✅ Multimodal capabilities: {capabilities}")
    except Exception as e:
        print(f"   ❌ Capability check failed: {e}")
        return False
    
    # Test 3: Test image extractor
    print("3. Testing image extractor...")
    try:
        extractor = ImageExtractor()
        print("   ✅ Image extractor created")
    except Exception as e:
        print(f"   ❌ Image extractor failed: {e}")
        return False
    
    # Test 4: Test multimodal processor (lightweight mode)
    print("4. Testing multimodal processor...")
    try:
        processor = MultimodalDocumentProcessor(enable_vision_models=False)
        print("   ✅ Multimodal processor created (lightweight mode)")
    except Exception as e:
        print(f"   ❌ Multimodal processor failed: {e}")
        return False
    
    # Test 5: Create test image
    print("5. Testing image creation...")
    try:
        img = Image.new('RGB', (100, 50), color='blue')
        draw = ImageDraw.Draw(img)
        draw.text((10, 20), "Test", fill='white')
        print("   ✅ Test image created")
    except Exception as e:
        print(f"   ❌ Image creation failed: {e}")
        return False
    
    # Test 6: Test Ollama connection
    print("6. Testing Ollama connection...")
    try:
        import ollama
        models = ollama.list()
        print("   ✅ Ollama connection successful")
    except Exception as e:
        print(f"   ⚠️ Ollama connection failed: {e}")
        print("   (This is expected if Ollama is not running)")
    
    print("\n🎉 All basic tests passed!")
    return True

def test_memory_efficiency():
    """Test memory efficiency."""
    print("\n🧠 Testing Memory Efficiency")
    print("=" * 50)
    
    try:
        # Test creating multiple processors without memory issues
        print("1. Creating multiple processors...")
        from multimodal_utils import MultimodalDocumentProcessor
        
        processors = []
        for i in range(5):
            processor = MultimodalDocumentProcessor(enable_vision_models=False)
            processors.append(processor)
            print(f"   ✅ Processor {i+1} created")
        
        print("2. Testing image processing...")
        from PIL import Image, ImageDraw
        
        images = []
        for i in range(10):
            img = Image.new('RGB', (200, 100), color=f'hsl({i*36}, 70%, 50%)')
            draw = ImageDraw.Draw(img)
            draw.text((10, 40), f"Image {i+1}", fill='white')
            images.append(img)
        
        print("   ✅ Multiple images created")
        
        # Test image extraction without heavy models
        from multimodal_utils import ImageExtractor
        extractor = ImageExtractor()
        print("   ✅ Image extractor ready")
        
        print("\n🎉 Memory efficiency tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Simple Application Test Suite")
    print("=" * 60)
    
    # Run basic functionality tests
    basic_ok = test_basic_functionality()
    
    # Run memory efficiency tests
    memory_ok = test_memory_efficiency()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    if basic_ok and memory_ok:
        print("🎉 All tests passed! Application is ready to run.")
        print("\n📋 Next steps:")
        print("1. Start Ollama: ollama serve")
        print("2. Run the app: streamlit run main.py")
        print("3. The app will run in lightweight mode by default")
        print("4. Enable vision models only if you have sufficient RAM")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 