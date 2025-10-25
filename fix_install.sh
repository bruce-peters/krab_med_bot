#!/bin/bash
# Fix installation issues for Python 3.12+

echo "Fixing PyTorch installation for Python 3.12..."
echo ""

echo "[1/3] Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "[2/3] Installing core dependencies..."
pip install fastapi uvicorn[standard] pydantic pydantic-settings httpx python-dotenv python-multipart aiofiles

echo ""
echo "[3/3] Installing AI dependencies..."
pip install openai anthropic spacy transformers openai-whisper TTS langchain

echo ""
echo "Installation complete!"
echo ""
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

echo ""
echo "Verifying installation..."
python check_install.py
