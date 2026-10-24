"""Entrypoint: run the IdeaCanvas backend.

    python run.py            # http://localhost:8000
    GEMINI_API_KEY=... python run.py

Set GEMINI_API_KEY (or Vertex/Enterprise env vars) to use the real Antigravity
SDK; without it the backend runs in placeholder mode so the prototype still works.
"""
import uvicorn

from backend.server import app


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
