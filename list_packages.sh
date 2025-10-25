#!/bin/bash

echo "========================================"
echo "Installed Python Packages"
echo "========================================"
echo ""

# Check if virtual environment is activated
python -c "import sys; print('Virtual Env:', 'YES' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 'NO')"
echo ""

echo "Core Packages:"
echo "--------------"
pip show fastapi uvicorn pydantic httpx 2>/dev/null

echo ""
echo "AI Packages:"
echo "------------"
pip show openai anthropic spacy 2>/dev/null

echo ""
echo "All Installed Packages:"
echo "-----------------------"
pip list

echo ""
echo "To save this list: pip freeze > installed_packages.txt"
