"""
AlveoSage Application Entrypoint.

Supports `flask run`, WSGI servers (Gunicorn, Waitress), and direct python execution.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.routes import create_app, load_environment_variables

# Load environment configuration from .env if present
load_environment_variables()

# Global application instance for Flask CLI and WSGI servers
app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
