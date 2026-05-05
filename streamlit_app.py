"""Root Streamlit entrypoint for local and Streamlit Cloud deployment.

The app implementation lives in app/streamlit_app.py. Keeping this wrapper at
the repository root makes Streamlit Community Cloud and the local Deploy button
detect the GitHub-backed app more reliably.
"""

from app.streamlit_app import *  # noqa: F401,F403

