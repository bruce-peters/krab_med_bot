#!/bin/bash
# Quick start script for Linux/Mac

echo "========================================"
echo "Krab Med Bot - Starting Server"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run setup.sh first"
    exit 1
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Starting FastAPI server..."
echo "Server will be available at: http://localhost:5000"
echo "API Documentation at: http://localhost:5000/docs"
echo ""
echo "Press CTRL+C to stop the server"
echo ""

uvicorn server.main:app --reload --host 0.0.0.0 --port 5000
