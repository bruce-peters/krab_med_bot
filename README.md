# krab_med_bot

Python backend and voice client for an AI medicine box for elderly users. A FastAPI server keeps the medication schedule, drives the box hardware (or a mock of it), runs a short AI check-in conversation when medication is taken, turns speech into text and replies into speech, and logs symptoms, which it can export as one JSON summary for a dashboard. `client_voice.py` is a microphone client that talks to the server out loud.

The ESP32 firmware and serial bridge for the physical box are in [krab_med_hardware](https://github.com/bruce-peters/krab_med_hardware).

More detail: [INSTALL.md](INSTALL.md) (full install guide), [RUN.md](RUN.md) (ways to run the server), [Implementation.md](Implementation.md) (design plan), [codebase_documentation.md](codebase_documentation.md) (architecture and API reference).

## Run it

### Prerequisites
- Python 3.12 (tested). Python 3.13+ does not work with the pinned packages, which have no prebuilt wheels for it.
- For the voice client only: a microphone and the PortAudio library (Linux: `sudo apt install portaudio19-dev`; macOS: `brew install portaudio`).

### 1. Install and configure
```bash
git clone https://github.com/bruce-peters/krab_med_bot.git
cd krab_med_bot
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # replaces the .env checked into the repo
```

To run with no API keys and no hardware, set these in `.env`:
```
HARDWARE_MODE=mock
AI_PROVIDER=mock
STT_PROVIDER=mock
TTS_PROVIDER=mock
```

### 2. Start the server
```bash
uvicorn server.main:app --reload --host 0.0.0.0 --port 5000
```
Open http://localhost:5000/docs for the interactive API docs, or try:
```bash
curl http://localhost:5000/health
curl -X POST http://localhost:5000/api/voice/interact \
  -H "Content-Type: application/json" -d '{"text_input": "I slept well"}'
curl -X POST http://localhost:5000/api/hardware/dispense \
  -H "Content-Type: application/json" \
  -d '{"compartment": 1, "medication_id": "550e8400-e29b-41d4-a716-446655440000"}'
```

### 3. Run the tests
```bash
pip install pytest
python -m pytest tests
python check_install.py    # lists installed packages
```

### 4. Voice client (optional, needs a microphone)
With the server running in another terminal:
```bash
pip install -r requirements_client.txt
python client_voice.py
```
It listens on the default microphone, sends the audio to `http://localhost:5000/api/voice/interact`, and plays the spoken reply.

## Configuration (`.env`)

| Variable | Values | Notes |
|---|---|---|
| `HARDWARE_MODE` | `mock`, `production` | `production` sends HTTP commands to a hardware controller (default `http://localhost:8080`, or `hardware/hardware_config.json` if present) |
| `AI_PROVIDER` | `mock`, `openai`, `anthropic`, `gemini`, `ollama` | Chat model for the check-in conversation |
| `LLM_MODEL` | e.g. `gemini-2.5-flash`, `gpt-4-turbo-preview` | Must match `AI_PROVIDER` |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` | your keys | Only the one for the providers you pick is needed |
| `STT_PROVIDER` | `mock`, `openai`, `gemini`, `whisper_local` | Speech-to-text. `openai` uses `whisper-1` |
| `TTS_PROVIDER` | `mock`, `openai`, `local`, `elevenlabs` | Text-to-speech. `local` uses pyttsx3 (on Linux it needs `espeak`) |

Never commit a real key. `.env.example` holds placeholders only.

## Project structure

```
server/
├── main.py            # FastAPI app, startup, /health
├── config.py          # Settings loaded from .env
├── routes/            # medication, hardware, health_data, ai_conversation, voice, data_export
├── ai/                # LLM client, conversation manager, symptom analyzer, speech (STT/TTS)
├── controllers/       # HTTP hardware interface + mock hardware
├── models/schemas.py  # Pydantic models
└── utils/             # JSON file storage, logging
client_voice.py        # Microphone client for /api/voice/interact
data/                  # JSON storage (sample data from development)
tests/                 # pytest model tests
```

## Notes and limitations

- **Tested here** (Linux, Python 3.12): `pip install -r requirements.txt`, `py_compile` on every file, `python -m pytest tests` (12 passed), and the server in mock mode (`/health`, `/docs`, starting and continuing an AI conversation, `/api/voice/interact`, `/api/hardware/dispense`, `/api/data/export/patient-data`). `requirements_client.txt` installs (after `portaudio19-dev`), and `client_voice.py` runs until it needs a microphone, which this machine does not have. Real AI providers, speech providers and hardware were not tested.
- `GET /api/medications/` and `GET /api/medications/next` return HTTP 500 with the bundled `data/medication_schedule.json`. That file uses the newer `box_index`/`frequency` medication format, while these two endpoints still read the older `compartment`/`scheduled_time` format.
- `STT_PROVIDER=gemini` uses the hard-coded model `gemini-2.0-flash-exp` in `server/ai/speech.py`, an experimental model that Google may no longer serve. `STT_PROVIDER=openai` avoids it.
- `HARDWARE_MODE=production` does not yet match the serial bridge in krab_med_hardware: the server expects `{"status": "success"}` replies and `GET /servo/status`, `/led/blink`, `/led/clear` endpoints, which the bridge does not provide. The bridge's other direction does work with the server in mock mode: it forwards box button presses to `POST /api/hardware/dispense` (on port 5800, so start the server with `--port 5800` or edit `DISPENSE_URL` in the bridge).
