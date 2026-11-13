"""
AlveoSage Application Launcher.

Starts the Flask backend server powering the medical AI web dashboard and REST API.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.routes import create_app, load_environment_variables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("alveosage_main")


def main():
    # Load .env variables before server initialization
    load_environment_variables()

    parser = argparse.ArgumentParser(description="Launch AlveoSage Medical AI Application.")
    parser.add_argument("--host", type=str, default=None, help="Server host binding (e.g. 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Server port (e.g. 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    # Load default app config
    config_file = PROJECT_ROOT / "config" / "app_config.json"
    with open(config_file, "r", encoding="utf-8") as f:
        app_config = json.load(f)

    host = args.host or app_config.get("server", {}).get("host", "127.0.0.1")
    port = args.port or app_config.get("server", {}).get("port", 5000)
    debug = args.debug or app_config.get("server", {}).get("debug", False)

    logger.info("Starting %s server at http://%s:%d (debug=%s)", app_config.get("app_name", "AlveoSage"), host, port, debug)
    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
