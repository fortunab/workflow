@echo off
call .venv\Scripts\activate
uvicorn app.api:app --host 0.0.0.0 --port 8000
pause
