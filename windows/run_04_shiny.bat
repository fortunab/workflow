@echo off
call .venv\Scripts\activate
shiny run shiny_app\app.py --reload --port 8001
pause
