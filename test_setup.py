#!/usr/bin/env python3
"""
Test script to verify the setup is working correctly.
"""

import sys
import subprocess

def test_imports():
    """Test if all required packages can be imported."""
    packages = [
        'streamlit',
        'langchain',
        'langchain_community',
        'chromadb',
        'fastembed',
        'pypdf',
        'pymupdf'
    ]
    
    print("Testing package imports...")
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            return False
    return True

def test_ollama():
    """Test if Ollama is accessible."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is accessible")
            return True
        else:
            print("❌ Ollama is not accessible")
            return False
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return False

def main():
    print("🧠 Testing Local RAG Web App Setup")
    print("=" * 40)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    print()
    
    # Test Ollama
    if not test_ollama():
        success = False
    
    print()
    
    if success:
        print("🎉 All tests passed! Your setup is ready.")
        print()
        print("To start the app, run:")
        print("  source venv/bin/activate")
        print("  streamlit run main.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
