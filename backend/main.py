"""
Shim to maintain compatibility with existing 'uvicorn backend.main:app' command.
Delegates to the refactored 'backend/app/main.py'.
"""
from backend.app.main import app
