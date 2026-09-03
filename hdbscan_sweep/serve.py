#!/usr/bin/env python3
"""Start a local web server and open the HDBSCAN sweep in the browser."""
import http.server, webbrowser, threading, os, socket
from pathlib import Path

os.chdir(Path(__file__).parent)
PORT = 8765

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args): pass  # suppress request logs

def find_port(start=8765):
    for p in range(start, start + 20):
        try:
            s = socket.socket(); s.bind(('', p)); s.close(); return p
        except OSError: pass
    return start

PORT = find_port()
server = http.server.HTTPServer(('localhost', PORT), Handler)
url = f'http://localhost:{PORT}'
print(f'Serving HDBSCAN sweep at {url}')
print('Press Ctrl+C to stop.')
threading.Timer(0.5, lambda: webbrowser.open(url)).start()
try:
    server.serve_forever()
except KeyboardInterrupt:
    print('\nServer stopped.')
