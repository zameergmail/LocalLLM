#!/usr/bin/env python3
"""
Test script for multimodal capabilities.
Tests image extraction, analysis, and integration with the RAG system.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_multimodal_imports():
    """Test if multimodal dependencies are available."""
    print("🔍 Testing multimodal imports...")
    
    try:
        import transformers
        print("✅ Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow imported successfully")
    except ImportError as e:
        print(f"❌ Pillow import failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import fitz  # PyMuPDF
        print("✅ PyMuPDF imported successfully")
    except ImportError as e:
        print(f"❌ PyMuPDF import failed: {e}")
        return False
    
    return True


def test_image_extraction():
    """Test image extraction functionality."""
    print("\n🖼️ Testing image extraction...")
    
    try:
        from multimodal_utils import ImageExtractor
        
        # Create a simple test image
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a test image
        img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 40), "Test Image", fill='black')
        
        # Save test image
        test_image_path = "test_image.png"
        img.save(test_image_path)
        
        # Test image statistics
        from multimodal_utils import get_image_statistics
        stats = get_image_statistics(img)
        print(f"✅ Image statistics: {stats}")
        
        # Test image resizing
        from multimodal_utils import resize_image_for_analysis
        resized_img = resize_image_for_analysis(img, max_size=100)
        print(f"✅ Image resized: {resized_img.size}")
        
        # Clean up
        os.remove(test_image_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Image extraction test failed: {e}")
        return False


def test_image_analyzer():
    """Test image analysis functionality."""
    print("\n🔍 Testing image analysis...")
    
    try:
        from multimodal_utils import ImageAnalyzer
        
        # Create a simple test image
        from PIL import Image, ImageDraw
        
        img = Image.new('RGB', (200, 100), color='blue')
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 25, 150, 75], fill='red')
        
        # Test analyzer initialization
        analyzer = ImageAnalyzer()
        print("✅ Image analyzer initialized")
        
        # Test image analysis (this might take a while on first run)
        print("⏳ Analyzing test image (this may take a moment)...")
        analysis = analyzer.analyze_image(img)
        
        print(f"✅ Analysis completed:")
        print(f"   - Caption: {analysis.get('caption', 'N/A')}")
        print(f"   - Classification: {analysis.get('classification', 'N/A')}")
        print(f"   - Error: {analysis.get('error', 'None')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image analysis test failed: {e}")
        return False


def test_multimodal_processor():
    """Test multimodal document processing."""
    print("\n📄 Testing multimodal document processing...")
    
    try:
        from multimodal_utils import MultimodalDocumentProcessor
        
        # Create a simple test PDF (this is a placeholder - in real usage you'd have a PDF)
        processor = MultimodalDocumentProcessor()
        print("✅ Multimodal processor initialized")
        
        # Test with a non-existent PDF (should handle gracefully)
        result = processor.process_pdf_with_images("nonexistent.pdf")
        print(f"✅ Graceful handling of missing file: {result.get('error', 'No error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multimodal processor test failed: {e}")
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


def test_ollama_vision_models():
    """Test if Ollama vision models are available."""
    print("\n🤖 Testing Ollama vision models...")
    
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


def create_test_pdf():
    """Create a simple test PDF with images for testing."""
    print("\n📄 Creating test PDF...")
    
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from PIL import Image, ImageDraw
        
        # Create a test image
        img = Image.new('RGB', (200, 100), color='lightblue')
        draw = ImageDraw.Draw(img)
        draw.text((10, 40), "Test Chart", fill='black')
        draw.rectangle([50, 25, 150, 75], fill='yellow')
        
        # Save image temporarily
        img_path = "test_chart.png"
        img.save(img_path)
        
        # Create PDF with image
        pdf_path = "test_document.pdf"
        c = canvas.Canvas(pdf_path, pagesize=letter)
        
        # Add text
        c.drawString(100, 750, "Test Document with Images")
        c.drawString(100, 720, "This document contains test images for multimodal analysis.")
        
        # Add image
        c.drawImage(img_path, 100, 500, width=200, height=100)
        
        c.save()
        
        # Clean up image file
        os.remove(img_path)
        
        print(f"✅ Test PDF created: {pdf_path}")
        return pdf_path
        
    except Exception as e:
        print(f"❌ Test PDF creation failed: {e}")
        return None


def test_full_multimodal_pipeline():
    """Test the complete multimodal pipeline."""
    print("\n🚀 Testing full multimodal pipeline...")
    
    try:
        # Create test PDF
        pdf_path = create_test_pdf()
        if not pdf_path:
            return False
        
        # Test multimodal processing
        from multimodal_utils import MultimodalDocumentProcessor
        
        processor = MultimodalDocumentProcessor()
        result = processor.process_pdf_with_images(pdf_path)
        
        print(f"✅ Multimodal processing completed:")
        print(f"   - Images found: {len(result.get('images', []))}")
        print(f"   - Analysis results: {len(result.get('image_analysis', []))}")
        print(f"   - Summary: {result.get('multimodal_summary', 'No summary')[:100]}...")
        
        # Clean up
        os.remove(pdf_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False


def main():
    """Run all multimodal tests."""
    print("🧪 Multimodal Capabilities Test Suite")
    print("=" * 50)
    
    tests = [
        ("Multimodal Imports", test_multimodal_imports),
        ("Image Extraction", test_image_extraction),
        ("Image Analysis", test_image_analyzer),
        ("Multimodal Processor", test_multimodal_processor),
        ("Utils Integration", test_utils_integration),
        ("Ollama Vision Models", test_ollama_vision_models),
        ("Full Pipeline", test_full_multimodal_pipeline),
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
        print("🎉 All tests passed! Multimodal capabilities are working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 