"""Las pruebas importan los módulos del dashboard como lo hace app.py (desde dashboard/)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
