"""Root Streamlit entrypoint for local and Streamlit Cloud deployment.

The app implementation lives in app/streamlit_app.py. Keeping this wrapper at
the repository root makes Streamlit Community Cloud and the local Deploy button
detect the GitHub-backed app more reliably.
"""

from __future__ import annotations

import importlib
import sys


MODULE_NAME = "app.streamlit_app"

if MODULE_NAME in sys.modules:
    importlib.reload(sys.modules[MODULE_NAME])
else:
    importlib.import_module(MODULE_NAME)
