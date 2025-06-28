#!/bin/bash

# 🧠 Local RAG Web App Setup Script for macOS
# This script automates the installation process for the Local RAG Web App

set -e  # Exit on any error

echo "🧠 Setting up Local RAG Web App..."
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is designed for macOS. Please use the manual installation instructions for other operating systems."
    exit 1
fi

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    print_warning "Homebrew is not installed. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    print_success "Homebrew installed successfully!"
else
    print_success "Homebrew is already installed"
fi

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_success "Python $PYTHON_VERSION is installed"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    print_status "Installing Ollama..."
    brew install ollama
    print_success "Ollama installed successfully!"
else
    print_success "Ollama is already installed"
fi

# Create virtual environment
print_status "Creating Python virtual environment..."
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Removing and recreating..."
    rm -rf venv
fi

python3 -m venv venv
print_success "Virtual environment created"

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
print_status "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "Python dependencies installed"
else
    print_error "requirements.txt not found. Please make sure you're in the correct directory."
    exit 1
fi

# Create necessary directories
print_status "Creating project directories..."
mkdir -p docs chroma
print_success "Project directories created"

# Check if Ollama is running
print_status "Checking Ollama status..."
if pgrep -x "ollama" > /dev/null; then
    print_success "Ollama is running"
else
    print_warning "Ollama is not running. Starting Ollama..."
    ollama serve &
    sleep 3
    print_success "Ollama started"
fi

# Pull a default model
print_status "Pulling default model (mistral)..."
if ollama list | grep -q "mistral"; then
    print_success "Mistral model is already available"
else
    print_status "Downloading Mistral model (this may take a few minutes)..."
    ollama pull mistral
    print_success "Mistral model downloaded"
fi

# Create a simple test script
print_status "Creating test script..."
cat > test_setup.py << 'EOF'
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
EOF

print_success "Test script created"

# Make test script executable
chmod +x test_setup.py

# Run the test
print_status "Running setup tests..."
python test_setup.py

print_success "Setup completed successfully!"
echo
echo "🎉 Your Local RAG Web App is ready!"
echo
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo
echo "2. Start the Streamlit app:"
echo "   streamlit run main.py"
echo
echo "3. Open your browser and go to:"
echo "   http://localhost:8501"
echo
echo "4. Upload a PDF or Markdown file and start chatting!"
echo
echo "For more information, see the README.md file."
echo
print_success "Happy chatting with your documents! 🧠✨" 