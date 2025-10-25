@echo off
REM filepath: /c:/Users/bruce/Projects/krab_med_bot/start.bat
REM Quick start script for Windows

echo ========================================
echo Krab Med Bot - Starting Server
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting FastAPI server...
echo Server will be available at: http://localhost:5000
echo API Documentation at: http://localhost:5000/docs
echo.
echo Press CTRL+C to stop the server
echo.

uvicorn server.main:app --reload --host 0.0.0.0 --port 5000

pause