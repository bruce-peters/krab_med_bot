
#!/bin/bash
# Linux/Mac Installation Script for Krab Med Bot

echo "========================================"
echo "Krab Med Bot - Automated Setup"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ first"
    exit 1
fi

echo "[1/7] Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

echo "[2/7] Activating virtual environment..."
source venv/bin/activate

echo "[3/7] Upgrading pip..."
python -m pip install --upgrade pip

echo "[4/7] Installing Python dependencies..."
echo "This may take 5-15 minutes..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo "[5/7] Downloading spaCy language model..."
python -m spacy download en_core_web_sm

echo "[6/7] Creating required directories..."
mkdir -p data/conversations
mkdir -p data/voice_recordings
mkdir -p logs

echo "[7/7] Verifying installation..."
python -c "import fastapi; import openai; import spacy; print('✓ Installation successful!')"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To start the server:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run server: uvicorn server.main:app --reload"
echo "  3. Open browser: http://localhost:5000/docs"
echo ""
echo "Remember to update your OpenAI API key in .env file!"
echo ""