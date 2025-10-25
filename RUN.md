# How to Run Krab Med Bot

## Quick Start

### 1. Activate Virtual Environment

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

You should see `(venv)` at the beginning of your prompt.

### 2. Run the Server

**Method 1: Using uvicorn directly (Recommended for Development)**

```bash
uvicorn server.main:app --reload --host 0.0.0.0 --port 5000
```

**Method 2: Using Python**

```bash
python -m uvicorn server.main:app --reload --host 0.0.0.0 --port 5000
```

**Method 3: Running the main.py directly**

```bash
python server/main.py
```

### 3. Access the Application

Once the server starts, you'll see:

```
INFO:     Uvicorn running on http://0.0.0.0:5000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Open these URLs in your browser:**

- **API Documentation (Swagger UI):** http://localhost:5000/docs
- **Alternative Documentation (ReDoc):** http://localhost:5000/redoc
- **Root Endpoint:** http://localhost:5000/
- **Health Check:** http://localhost:5000/health
- **Test Endpoint:** http://localhost:5000/api/test

## Command Options Explained

```bash
uvicorn server.main:app --reload --host 0.0.0.0 --port 5000
```

- `server.main:app` - Path to FastAPI app instance
- `--reload` - Auto-reload on code changes (development only)
- `--host 0.0.0.0` - Listen on all network interfaces
- `--port 5000` - Port number (default: 8000)

## Additional Options

### Run on Different Port

```bash
uvicorn server.main:app --reload --port 8000
```

### Production Mode (No Auto-reload)

```bash
uvicorn server.main:app --host 0.0.0.0 --port 5000
```

### Production with Multiple Workers

```bash
uvicorn server.main:app --host 0.0.0.0 --port 5000 --workers 4
```

### With Custom Log Level

```bash
uvicorn server.main:app --reload --log-level debug
```

## Testing the API

### Using cURL (Command Line)

**Test root endpoint:**

```bash
curl http://localhost:5000/
```

**Test health check:**

```bash
curl http://localhost:5000/health
```

**Test API endpoint:**

```bash
curl http://localhost:5000/api/test
```

### Using Browser

Just open http://localhost:5000/docs and you can test all endpoints interactively!

## Troubleshooting

### Issue: "Address already in use"

**Solution:** Port 5000 is being used by another application

```bash
# Use a different port
uvicorn server.main:app --reload --port 8000

# Or find and kill the process using port 5000 (Windows)
netstat -ano | findstr :5000
taskkill /PID <process_id> /F

# Or find and kill the process (Linux/Mac)
lsof -i :5000
kill -9 <PID>
```

### Issue: "Module not found: server"

**Solution:** Make sure you're in the project root directory

```bash
cd /c:/Users/bruce/Projects/krab_med_bot
python -m uvicorn server.main:app --reload
```

### Issue: "No module named 'pydantic_settings'"

**Solution:** Virtual environment not activated or dependencies not installed

```bash
# Activate venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Issue: FastAPI won't start

**Solution:** Check Python version and installation

```bash
python --version  # Should be 3.8+
python check_install.py  # Run installation checker
```

## Stop the Server

Press `CTRL+C` in the terminal where the server is running.

## Running in Background (Linux/Mac)

```bash
# Start in background
nohup uvicorn server.main:app --host 0.0.0.0 --port 5000 > server.log 2>&1 &

# Check if running
ps aux | grep uvicorn

# Stop background process
kill <PID>
```

## Running as Windows Service

For production deployment on Windows, consider using:

- NSSM (Non-Sucking Service Manager)
- Or Task Scheduler

## Next Steps

1. ✅ Server is running
2. 📖 Read the API documentation at http://localhost:5000/docs
3. 🔨 Start implementing features from Implementation.md
4. 🧪 Write tests in the `tests/` directory
5. 🚀 Deploy to production when ready
