"""Evaluator UI on Laptop 2.

Serves static UI on port 8080 (or UI_PORT env var).
Owner: Member A & C
"""
import http.server
import os
from pathlib import Path
import socketserver

PORT = int(os.environ.get("UI_PORT", 8080))


def main():
    ui_dir = Path(__file__).resolve().parent
    os.chdir(ui_dir)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("0.0.0.0", PORT), handler) as httpd:
        print(f"PathSnap Evaluator UI running on http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down UI server.")


if __name__ == "__main__":
    main()
