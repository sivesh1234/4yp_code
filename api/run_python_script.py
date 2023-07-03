from http.server import BaseHTTPRequestHandler
from os import environ
import subprocess

class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type','text/plain')
        self.end_headers()

        # Run the Python script and get the output
        output = subprocess.check_output(["python", "your_script.py"], text=True)

        self.wfile.write(output.encode())
        return