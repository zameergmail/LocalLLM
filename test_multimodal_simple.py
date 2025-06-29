#!/usr/bin/env python3
"""
Simplified test script for multimodal capabilities.
Tests core functionality without heavy vision models.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test basic multimodal imports."""
    print("🔍 Testing basic multimodal imports...")
    
    try:
        import fitz  # PyMuPDF
        print("✅ PyMuPDF imported successfully")
    except ImportError as e:
        print(f"❌ PyMuPDF import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow imported successfully")
    except ImportError as e:
        print(f"❌ Pillow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    return True


def test_image_processing():
    """Test basic image processing functionality."""
    print("\n🖼️ Testing image processing...")
    
    try:
        from PIL import Image, ImageDraw
        
        # Create a test image
        img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 40), "Test Image", fill='black')
        
        print(f"✅ Test image created: {img.size}")
        
        # Test image operations
        resized = img.resize((100, 50))
        print(f"✅ Image resized: {resized.size}")
        
        # Test saving and loading
        test_path = "test_image.png"
        img.save(test_path)
        loaded_img = Image.open(test_path)
        print(f"✅ Image saved and loaded: {loaded_img.size}")
        
        # Clean up
        os.remove(test_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        return False


def test_pdf_processing():
    """Test PDF processing functionality."""
    print("\n📄 Testing PDF processing...")
    
    try:
        import fitz
        
        # Create a simple test PDF
        doc = fitz.open()
        page = doc.new_page()
        
        # Add some text
        page.insert_text((50, 50), "Test PDF Document")
        
        # Save PDF
        test_pdf_path = "test_document.pdf"
        doc.save(test_pdf_path)
        doc.close()
        
        print(f"✅ Test PDF created: {test_pdf_path}")
        
        # Test reading PDF
        doc = fitz.open(test_pdf_path)
        page = doc.load_page(0)
        text = page.get_text()
        print(f"✅ PDF text extracted: {text[:50]}...")
        
        # Test image extraction (even if no images)
        images = page.get_images()
        print(f"✅ Images found: {len(images)}")
        
        doc.close()
        
        # Clean up
        os.remove(test_pdf_path)
        
        return True
        
    except Exception as e:
        print(f"❌ PDF processing test failed: {e}")
        return False


def test_multimodal_utils():
    """Test multimodal utilities without heavy models."""
    print("\n🔧 Testing multimodal utilities...")
    
    try:
        # Test if we can import the module
        from multimodal_utils import ImageExtractor, encode_image_to_base64, decode_base64_to_image
        
        print("✅ Multimodal utilities imported successfully")
        
        # Test image extractor
        extractor = ImageExtractor()
        print("✅ Image extractor initialized")
        
        # Test image encoding/decoding
        from PIL import Image, ImageDraw
        
        img = Image.new('RGB', (100, 50), color='blue')
        encoded = encode_image_to_base64(img)
        decoded = decode_base64_to_image(encoded)
        
        print(f"✅ Image encoding/decoding: {decoded.size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multimodal utilities test failed: {e}")
        return False


def test_utils_integration():
    """Test integration with utils module."""
    print("\n🔧 Testing utils integration...")
    
    try:
        from utils import check_multimodal_capabilities, get_image_analysis_summary
        
        # Test capability check
        capabilities = check_multimodal_capabilities()
        print(f"✅ Multimodal capabilities: {capabilities}")
        
        # Test image analysis summary (with non-existent file)
        summary = get_image_analysis_summary("nonexistent.pdf")
        print(f"✅ Image analysis summary: {summary}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utils integration test failed: {e}")
        return False


def test_ollama_models():
    """Test Ollama model availability."""
    print("\n🤖 Testing Ollama models...")
    
    try:
        import ollama
        
        # List available models
        models = ollama.list()
        model_names = [model['name'] for model in models['models']]
        
        print(f"✅ Available Ollama models: {model_names}")
        
        # Check for vision-capable models
        vision_models = [name for name in model_names if 'vision' in name.lower() or 'llava' in name.lower()]
        
        if vision_models:
            print(f"✅ Vision models found: {vision_models}")
        else:
            print("⚠️ No vision models found. Consider pulling a vision model:")
            print("   ollama pull llava")
            print("   ollama pull bakllava")
        
        return True
        
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return False


def test_streamlit_app():
    """Test if the Streamlit app can be imported."""
    print("\n🌐 Testing Streamlit app...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        # Test if main app can be imported
        import main
        print("✅ Main app imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit app test failed: {e}")
        return False


def main():
    """Run simplified multimodal tests."""
    print("🧪 Simplified Multimodal Capabilities Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Image Processing", test_image_processing),
        ("PDF Processing", test_pdf_processing),
        ("Multimodal Utils", test_multimodal_utils),
        ("Utils Integration", test_utils_integration),
        ("Ollama Models", test_ollama_models),
        ("Streamlit App", test_streamlit_app),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Core multimodal capabilities are working.")
        print("\n📋 Next steps:")
        print("1. Start Ollama: ollama serve")
        print("2. Run the app: streamlit run main.py")
        print("3. Upload a PDF with images to test full functionality")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        print("\n💡 For full vision model testing, consider:")
        print("- Using a machine with more RAM")
        print("- Installing smaller vision models")
        print("- Testing with simpler image analysis")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 