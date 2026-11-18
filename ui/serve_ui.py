"""Serve the static UI on Laptop 2:  python ui/serve_ui.py  -> http://localhost:8080  (set the API URL in the page to http://<laptop1-ip>:8000)"""
import http.server, socketserver, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
with socketserver.TCPServer(("0.0.0.0", 8080), http.server.SimpleHTTPRequestHandler) as s:
    print("UI on http://0.0.0.0:8080"); s.serve_forever()
