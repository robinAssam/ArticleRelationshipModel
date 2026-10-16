"""Streamlit Community Cloud entry point.

Community Cloud installs the requirements.txt that sits next to the entry
file, so this folder carries a slim, CPU-only dependency list for hosting.
The app itself is the repository's app.py, run unchanged.
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[2] / "app.py"), run_name="__main__")
