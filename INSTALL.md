# Installation Guide - Krab Med Bot

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for version control)
- 2-4 GB free disk space (for ML models)

## Step-by-Step Installation

### 1. Navigate to Project Directory

```bash
cd /c:/Users/bruce/Projects/krab_med_bot
```

### 2. Create Virtual Environment

**Windows:**

```bash
python -m venv venv
```

**Linux/Mac:**

```bash
python3 -m venv venv
```

### 3. Activate Virtual Environment

**Windows (Command Prompt):**

```bash
venv\Scripts\activate
```

**Windows (PowerShell):**

```bash
venv\Scripts\Activate.ps1
```

**Linux/Mac:**

```bash
source venv/bin/activate
```

You should see `(venv)` at the beginning of your command prompt.

### 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 5. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:

- FastAPI and Uvicorn (web server)
- Pydantic (data validation)
- httpx (async HTTP client)
- OpenAI SDK (AI integration)
- Anthropic SDK (Claude AI)
- spaCy (NLP)
- Transformers and PyTorch (ML models)
- And more...

**Note:** This may take 5-15 minutes depending on your internet connection.

### 6. Download spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 7. Create Required Directories

**Windows:**

```bash
mkdir data\conversations
mkdir data\voice_recordings
mkdir logs
```

**Linux/Mac:**

```bash
mkdir -p data/conversations
mkdir -p data/voice_recordings
mkdir -p logs
```

### 8. Verify Installation

```bash
python -c "import fastapi; import openai; import spacy; print('✓ All core packages installed successfully!')"
```

If you see the success message, you're good to go!

## Configuration

### 1. Set Up Environment Variables

The `.env` file is already created. Update these important values:

```bash
# Open .env in a text editor
notepad .env  # Windows
nano .env     # Linux/Mac
```

**Required changes for AI features:**

- Replace `your_openai_key_here` with your actual OpenAI API key
- Get API key from: https://platform.openai.com/api-keys

**For testing without AI:**

- Keep `AI_PROVIDER=mock`
- Keep `STT_PROVIDER=mock`
- Keep `TTS_PROVIDER=mock`

### 2. Test Configuration

```bash
python -c "from server.config import settings; print(f'App: {settings.app_name}'); print(f'Mode: {settings.hardware_mode}')"
```

## Running the Server

### Development Mode (with auto-reload)

```bash
uvicorn server.main:app --reload --host 0.0.0.0 --port 5000
```

### Production Mode

```bash
uvicorn server.main:app --host 0.0.0.0 --port 5000 --workers 4
```

### Access the API

Once running, open your browser to:

- **API Documentation:** http://localhost:5000/docs
- **Alternative Docs:** http://localhost:5000/redoc
- **Health Check:** http://localhost:5000/health

## Troubleshooting

### Issue: "command not found: python"

**Solution:** Try `python3` instead of `python`

### Issue: PyTorch installation fails

**Solution:** Install CPU-only version:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Permission denied on Windows PowerShell

**Solution:** Run this command first:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Module not found errors

**Solution:** Make sure virtual environment is activated and reinstall:

```bash
pip install -r requirements.txt --force-reinstall
```

### Issue: Port 5000 already in use

**Solution:** Use a different port:

```bash
uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

### Issue: Out of memory during installation

**Solution:** Install packages one at a time:

```bash
pip install fastapi uvicorn pydantic pydantic-settings
pip install httpx python-dotenv aiofiles
pip install openai anthropic
pip install spacy
python -m spacy download en_core_web_sm
```

## Optional: Install Without AI Dependencies

If you only want to test the basic API without AI features:

### Minimal Requirements File

Create `requirements-minimal.txt`:

```
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.0
pydantic-settings==2.1.0
httpx==0.26.0
python-dotenv==1.0.0
python-multipart==0.0.6
aiofiles==23.2.1
```

Install minimal version:

```bash
pip install -r requirements-minimal.txt
```

Set in `.env`:

```
HARDWARE_MODE=mock
AI_PROVIDER=mock
STT_PROVIDER=mock
TTS_PROVIDER=mock
```

## Next Steps

After successful installation:

1. **Test the API:**

   ```bash
   curl http://localhost:5000/health
   ```

2. **Explore API docs:**
   Open http://localhost:5000/docs in your browser

3. **Start implementing:**
   Follow the Implementation.md guide for building features

## Uninstallation

To completely remove the project:

```bash
# Deactivate virtual environment
deactivate

# Delete virtual environment
rm -rf venv  # Linux/Mac
rmdir /s venv  # Windows

# Delete downloaded models (optional)
rm -rf ~/.cache/huggingface
```

## Getting Help

- Check `Implementation.md` for detailed feature documentation
- Check `README.md` for project overview
- Review FastAPI docs: https://fastapi.tiangolo.com/
- Review OpenAI docs: https://platform.openai.com/docs/
