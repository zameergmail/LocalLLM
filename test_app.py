#!/usr/bin/env python3
"""
Test script to verify the Local RAG Web App setup.
"""

import sys
import subprocess
import os

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
    failed_imports = []
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def test_ollama():
    """Test if Ollama is accessible."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Ollama is accessible")
            # Print available models
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            
            if models:
                print(f"   Available models: {', '.join(models)}")
            else:
                print("   No models found. Run 'ollama pull mistral' to download a model.")
            return True
        else:
            print("❌ Ollama is not accessible")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Ollama command timed out")
        return False
    except FileNotFoundError:
        print("❌ Ollama not found. Install with 'brew install ollama'")
        return False
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return False

def test_directories():
    """Test if necessary directories exist."""
    directories = ['docs', 'chroma']
    
    print("Testing directories...")
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ {directory}/")
        else:
            print(f"⚠️  {directory}/ (will be created automatically)")

def test_environment():
    """Test environment setup."""
    print("Testing environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor}.{python_version.micro} (requires 3.8+)")
        return False
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Not running in virtual environment (recommended)")
    
    return True

def main():
    print("🧠 Testing Local RAG Web App Setup")
    print("=" * 40)
    
    success = True
    
    # Test environment
    if not test_environment():
        success = False
    
    print()
    
    # Test imports
    if not test_imports():
        success = False
    
    print()
    
    # Test directories
    test_directories()
    
    print()
    
    # Test Ollama
    if not test_ollama():
        success = False
    
    print()
    print("=" * 40)
    
    if success:
        print("🎉 All tests passed! Your setup is ready.")
        print()
        print("To start the app, run:")
        print("  source venv/bin/activate")
        print("  streamlit run main.py")
        print()
        print("The app will be available at: http://localhost:8501")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print()
        print("Common fixes:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Install Ollama: brew install ollama")
        print("3. Pull a model: ollama pull mistral")
        print("4. Activate virtual environment: source venv/bin/activate")
        sys.exit(1)

if __name__ == "__main__":
    main() 