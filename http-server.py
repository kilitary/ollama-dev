#!/usr/bin/env python3

import http.server
import socketserver
import ssl
import json
import urllib.parse
import sys
import threading
import argparse
import rich
import subprocess
import gpiozero
import random
import re
from signal import pause as pauser
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

contact_led = gpiozero.LED("GPIO23")
contact_led_ten = gpiozero.LED("GPIO24")
contact_led_ten_ten = gpiozero.LED("GPIO15")

# Icons for different log levels and events
ICONS = {
    'request': '📥',
    'response': '📤',
    'info': 'ℹ️',
    'success': '✅',
    'warning': '⚠️',
    'error': '❌',
    'http': '🔵',
    'https': '🔒',
    'server_start': '🚀',
    'server_stop': '🛑',
    'client': '[c]',
    'headers': '📋',
    'body': '📦',
    'method': '⚡',
    'path': '🛤️',
    'port': '🔌',
    'certificate': '📜',
    'key': '🔑',
    'thread': '🧵',
    'lock': '🔐',
    'hping3': '🎯',
    'random': '[r]',
    'jss': '🏷️',
    'timer': '⏱️',
    'debug': '🔍',
    "server": 'ผ(•̀_•́ผ)'
}


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Thread-safe TCP server using threads for each request"""
    allow_reuse_address = True
    daemon_threads = False  # Wait for threads to complete on shutdown


class HeaderDisplayHandler(http.server.BaseHTTPRequestHandler):
    # Make handler class-level variables thread-safe with locks
    _lock = threading.Lock()
    _request_count = 0
    _should_exit_after_one = False  # Flag to indicate if server should exit after one request
    _server_instance = None  # Reference to the server instance
    _request_id = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Thread-safe counter increment
        with self._lock:
            HeaderDisplayHandler._request_count += 1
            self._request_id = HeaderDisplayHandler._request_count

    @classmethod
    def set_exit_after_one(cls, should_exit):
        """Set whether the server should exit after handling one request"""
        cls._should_exit_after_one = should_exit

    @classmethod
    def set_server_instance(cls, server):
        """Set the server instance for shutdown access"""
        cls._server_instance = server

    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(f"{ICONS['server']} {format % args}")

    def _log_request_start(self, method, path, client_ip, client_port):
        """Log the start of a request with icons"""
        logger.info(f"{ICONS['request']} Request #{self._request_id} started")
        logger.info(f"  {ICONS['method']} Method: {method}")
        logger.info(f"  {ICONS['path']} Path: {path}")
        logger.info(f"  {ICONS['client']} Client: {client_ip}:{client_port}")
        logger.debug(f"  {ICONS['thread']} Thread: {threading.current_thread().name}")

    def _log_headers(self, headers):
        """Log request headers"""
        logger.debug(f"  {ICONS['headers']} Headers:")
        for key, value in headers.items():
            logger.debug(f"    {key}: {value}")

    def _log_body(self, body):
        """Log request body if present"""
        if body:
            logger.debug(f"  {ICONS['body']} Body length: {len(body)} bytes")
            logger.debug(f"    Body preview: {body[:200]}...")

    def _log_hping3(self, random_data, client_ip, client_port):
        """Log hping3 command execution"""
        logger.info(f"  {ICONS['hping3']} Executing hping3...")
        logger.debug(f"    {ICONS['random']} Random data: {random_data}")
        logger.debug(f"    Target: {client_ip}:{client_port}")

    def _log_response(self, jss_value, response_size):
        """Log response sending"""
        logger.info(f"  {ICONS['response']} Sending response")
        logger.info(f"    {ICONS['jss']} JSS value: {jss_value}")
        logger.info(f"    {ICONS['body']} Response size: {response_size} bytes")
        logger.info(f"{ICONS['success']} Request #{self._request_id} completed")

    def _log_error(self, error_msg):
        """Log errors"""
        logger.error(f"{ICONS['error']} Error: {error_msg}")

    def do_GET(self):
        self._handle_request()

    def do_POST(self):
        self._handle_request()

    def do_PUT(self):
        self._handle_request()

    def do_DELETE(self):
        self._handle_request()

    def _handle_request(self):
        # Get all headers
        headers = dict(self.headers)

        # Get request path
        path = self.path

        # Get request method
        method = self.command

        # For POST/PUT requests, read the body
        content_length = int(self.headers.get('Content-Length', 0))
        body = None
        if content_length > 0:
            body = self.rfile.read(content_length).decode('utf-8')

        # Extract client IP and port
        client_ip = self.client_address[0]
        client_port = self.client_address[1]


        # Generate a random HTTP Date header (RFC 1123 format)
        random_timestamp = datetime.now().timestamp() - random.randint(0, 31536000)  # up to 1 year ago
        random_date_str = datetime.fromtimestamp(random_timestamp).strftime('%a, %d %b %Y %H:%M:%S GMT')

        # Generate random HTTP/1.1 headers (3-10 headers)
        browser_headers = [# Safari safe headers with integer variations
        ('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'),
        ('Accept-Language', f'en-us'),
        ('Accept-Encoding', 'gzip, deflate'),
        ('User-Agent', f'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_{random.randint(1, 11119)}) AppleWebKit/605.1.{random.randint(1, 111150)} (KHTML, like Gecko) Version/14.{random.randint(0, 511111)} Safari/605.1.{random.randint(1, 5111110)}'),

        # Chrome safe headers with integer variations
        ('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8'),
        ('Accept-Language', 'en-US,en;q=0.9'),
        ('Accept-Encoding', 'gzip, deflate, br'),
        ('User-Agent', f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.{random.randint(4000, 500110)}.{random.randint(100, 20110)} Safari/537.36'),

        # Firefox safe headers with integer variations
        ('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'),
        ('Accept-Language', 'en-US,en;q=0.5'),
        ('Accept-Encoding', 'gzip, deflate'),
        ('User-Agent', f'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:{random.randint(80, 120)}.0) Gecko/20100101 Firefox/{random.randint(80, 121110)}.0'),

        # Edge safe headers with integer variations
        ('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'),
        ('Accept-Language', 'en-US,en;q=0.9'),
        ('Accept-Encoding', 'gzip, deflate, br'),
        ('User-Agent', f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.{random.randint(4000, 5011100)}.{random.randint(100, 20110)} Safari/537.36 Edg/91.0.{random.randint(500, 999)}.{random.randint(10, 99)}'),

        # pukfriz
        ('User-Agent', f'Mozilla/5.1 (Windows NT 13.43; Win65; x65) AppleWebKit/537.36 (KHTML, like pukfriz) Chem/91.0.{random.randint(4000, 5011100)}.{random.randint(100, 20110)} Zafaru/537.36 Ledg/91.0.{random.randint(500, 999)}.{random.randint(10, 99)}'),

        # Generic headers with integer variations
        ('Connection', 'keep-alive'),
        ('Cache-Control', f'max-age={random.randint(0, 3600)}'),
        ('Upgrade-Insecure-Requests', f'{random.randint(0, 2)}'),
        ('DNT', f'{random.randint(0, 2)}'),
        ('Sec-Fetch-Dest', 'document'),
        ('Sec-Fetch-Mode', 'navigate'),
        ('Sec-Fetch-Site', 'none'),
        ('Sec-Fetch-User', f'?{random.randint(1, 3)}')]

        # Randomly select 3-10 headers ensuring no conflicts
        num_headers = random.randint(4, 10)
        # Fix browser_groups to have valid indices within the browser_headers list
        browser_groups = {
            'safari': [0, 1, 2, 3],
            'chrome': [4, 5, 6, 7],
            'firefox': [8, 9, 10, 11],
            'edge': [12, 13, 14, 15],
            'generic': [16, 17, 18, 19, 20, 21, 22, 23],
            'pukfriz': [24]
        }

        # Select one browser group to prioritize (to avoid conflicts)
        browsers = ['safari', 'chrome', 'firefox', 'edge', 'pukfriz']
        primary_browser = random.choice(browsers)

        # Select headers: prioritize primary browser + generic headers
        selected_indices = []
        selected_indices.extend(random.sample(browser_groups[primary_browser],
                                            min(2, len(browser_groups[primary_browser]))))
        selected_indices.extend(random.sample(browser_groups['generic'],
                                            min(num_headers - len(selected_indices),
                                                len(browser_groups['generic']))))

        # Add remaining headers from any group if needed
        all_indices = [i for indices in browser_groups.values() for i in indices]
        remaining_slots = num_headers - len(selected_indices)
        if remaining_slots > 0:
            available_indices = [i for i in all_indices if i not in selected_indices]
            selected_indices.extend(random.sample(available_indices,
                                                min(remaining_slots, len(available_indices))))

        self.send_response(200)

        # Add 3 to 20 random Date headers
        for _ in range(random.randint(13, 69)):
            random_timestamp = datetime.now().timestamp() - random.randint(0, 31536000)  # up to 1 year ago
            random_date_str = datetime.fromtimestamp(random_timestamp).strftime('%a, %d %b %Y %H:%M:%S GMT')
            self.send_header('Date', random_date_str)


        # Add remaining headers from any group if needed
        all_indices = [i for indices in browser_groups.values() for i in indices]
        remaining_slots = num_headers - len(selected_indices)
        if remaining_slots > 0:
            available_indices = [i for i in all_indices if i not in selected_indices]
            selected_indices.extend(random.sample(available_indices,
                                                min(remaining_slots, len(available_indices))))

        # Add selected headers
        added_headers = set()
        for i in selected_indices:
            header_name, header_value = browser_headers[i]
            # Avoid duplicate headers
            if header_name.lower() not in added_headers:
                self.send_header(header_name, header_value)
                added_headers.add(header_name.lower())


        # # Randomize the order of the headers before sending
        # headers_list = list(self._headers_buffer)
        # random.shuffle(headers_list)
        # self._headers_buffer = headers_list


        # Log request start
        self._log_request_start(method, path, client_ip, client_port)
        self._log_headers(headers)
        self._log_body(body)

        # Generate random data for hping3
        random_data = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=10))
        rich.print(f'[server] Random data: {random_data}')
        # Use hping3 to send ACK+PSH packet with random data
        jss_value = None
        try:
            self._log_hping3(random_data, client_ip, client_port)

            hping3_cmd = [
                'hping3',
                '-P',  # PSH flag (we'll use this as part of ACK+PSH)
                '-A',  # ACK flag
                '-p', str(client_port),  # Destination port (client's source port)
                client_ip,
                '--data', random_data,
                '-c', '2'  # Send only one packet
            ]

            logger.debug(f"  {ICONS['debug']} Command: {' '.join(hping3_cmd)}")

            # Run hping3 and capture output
            result = subprocess.run(hping3_cmd, capture_output=True, text=True, timeout=10)

            logger.debug(f"  {ICONS['debug']} hping3 stdout: {result.stdout}")
            logger.debug(f"  {ICONS['debug']} hping3 stderr: {result.stderr}")

            # Calculate relative ID from hping3 output
            jss_value = self._calculate_relative_id(result.stdout, result.stderr, random_data)

            on_time = max(0.2, (float(jss_value) % 0.5) * 1.2)

            contact_led.blink(on_time=on_time, off_time=0.1, n=3)

            if int(jss_value) >= 100:
                contact_led_ten.blink(on_time=on_time, off_time=0.1, n=3)
            if int(jss_value) >= 200:
                contact_led_ten_ten.blink(on_time=on_time, off_time=0.1, n=3)

            logger.info(f"  {ICONS['jss']} Calculated JSS: {jss_value}")


        except subprocess.TimeoutExpired:
            self._log_error(f"hping3 command timed out for client {client_ip}:{client_port}")
            jss_value = f"TIMEOUT:{random_data}"
        except subprocess.CalledProcessError as e:
            self._log_error(f"hping3 command failed: {e}")
            jss_value = f"FAIL:{random_data}"
        except FileNotFoundError:
            self._log_error("hping3 not found - is it installed?")
            jss_value = f"NOTFOUND:{random_data}"
        except Exception as e:
            self._log_error(f"Unexpected error in hping3: {e}")
            jss_value = f"ERROR:{random_data}"
            tb = sys.exc_info()

            rich.print(e.with_traceback(tb))

        # Prepare response data
        response_data = {
            "method": method,
            "path": path,
            "headers": headers,
            "client_ip": client_ip,
            "client_port": client_port,
            "jss": f"JSS={jss_value}"
        }

        if body is not None:
            response_data["body"] = body

        # Convert to JSON for display
        response_json = f"\r\n\r\n<html><meta name='charset' content='utf-8'>{ICONS['client']} Client: {client_ip}:{client_port}"
        response_json += f" {ICONS['jss']} JSS Value: JSS={jss_value} "
        response_json += json.dumps(response_data, indent="<br/>")

        # Display the information
        response_json += f"{ICONS['random']} Random Data: {random_data}\n"

        logger.debug(f"  {ICONS['debug']} Full response: {response_json}")


        self.send_header('Date', random_date_str.strip())
        self.send_header('Content-rate', 'mean(sec)=0.6497, tg=off, put="async get", eth2=lookback')
        self.send_header('server', 'vary/0.9,q:6=?text=lain')
        self.send_header('Content-type', 'text/html')
        self.send_header('Content-length', str(len(response_json)))
        self.flush_headers()

        self.wfile.write(response_json.encode('utf-8'))

        # Log response
        self._log_response(jss_value, len(response_json))

        # If --once flag is set, shutdown the server after handling this request
        if HeaderDisplayHandler._should_exit_after_one:
            logger.info(f"{ICONS['server_stop']} --once flag set, shutting down server after handling request #{self._request_id}")
            if HeaderDisplayHandler._server_instance:
                # Run shutdown in a separate thread to avoid blocking
                shutdown_thread = threading.Thread(target=HeaderDisplayHandler._server_instance.shutdown)
                shutdown_thread.daemon = True
                shutdown_thread.start()

                shutdown_thread.join()
                sys.exit(0)



    def _calculate_relative_id(self, stdout, stderr, random_data):
        """Calculate relative ID from hping3 output"""
        try:
            prev_id = -1
            # Try to extract ID from hping3 output
            # Look for patterns like "id=" or "seq=" in the output
            combined_output = stdout + stderr
            logger.debug(f"  {ICONS['debug']} hping3 combined output:\n{combined_output}")
            relative_id = 0
            # Try to find any numeric ID in the output
            id_pattern = r'id=(\d+)'

            matches = re.findall(id_pattern, combined_output)
            if matches:
                logger.debug(f"  {ICONS['debug']} Found {len(matches)} ID matches")
                for match in matches:
                    logger.debug(f"    {ICONS['debug']} Match: {match}")
                    found_id = int(match)
                    if prev_id == -1:
                        prev_id = found_id
                    else:
                        relative_id = found_id - prev_id
                        prev_id = found_id
                    logger.debug(f"    {ICONS['debug']} Calculated relative ID: id={found_id}, relative={relative_id}")

            if relative_id > 0:
                logger.debug(f"  {ICONS['debug']} Final relative ID: {relative_id}")
                return str(relative_id)

            # If no pattern matches, use hash of random data
            hash_value = sum(ord(c) for c in random_data) % 10000
            logger.debug(f"  {ICONS['debug']} Using fallback hash: {hash_value}")
            return str(hash_value)

        except Exception as e:
            logger.error(f"  {ICONS['error']} Error calculating relative ID: {e}")
            # Fallback to simple hash of random data
            hash_value = sum(ord(c) for c in random_data) % 10000
            return str(hash_value)


def create_server(port, use_https=False, certfile=None, keyfile=None, exit_after_one=False):
    """Create and configure a server (HTTP or HTTPS)"""
    host = "0.0.0.0"

    # Use ThreadingTCPServer for thread-safe concurrent request handling
    httpd = ThreadingTCPServer((host, port), HeaderDisplayHandler)

    # Set the server instance for shutdown access
    HeaderDisplayHandler.set_server_instance(httpd)

    # Set exit after one flag if needed
    HeaderDisplayHandler.set_exit_after_one(exit_after_one)

    protocol_icon = ICONS['https'] if use_https else ICONS['http']

    if use_https:
        if not certfile or not keyfile:
            logger.info(f"{ICONS['certificate']} Generating self-signed certificate for HTTPS...")
            # Generate a self-signed certificate
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:4096",
                "-keyout", "key.pem", "-out", "cert.pem",
                "-days", "365", "-nodes", "-subj",
                "/C=US/ST=State/L=City/O=Organization/CN=localhost"
            ], check=True)
            certfile, keyfile = "cert.pem", "key.pem"
            logger.info(f"{ICONS['key']} Certificate generated: cert.pem, key.pem")

        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile=certfile, keyfile=keyfile)
        httpd.socket = ssl_context.wrap_socket(httpd.socket, server_side=True)
        protocol = "HTTPS"
    else:
        protocol = "HTTP"

    logger.info(f"{ICONS['server_start']} Starting {protocol} server on port {port}")
    logger.info(f"{ICONS['port']} Access the server at: {protocol.lower()}://{host}:{port}")
    logger.info(f"{ICONS['thread']} Using ThreadingTCPServer for thread-safe handling")
    if exit_after_one:
        logger.info(f"{ICONS['info']} Server will exit after handling one request (--once flag)")
    return httpd, protocol


def run_servers(http_port=8000, https_port=8443, certfile=None, keyfile=None):
    """Run both HTTP and HTTPS servers concurrently"""
    logger.info(f"{ICONS['server_start']} Initializing servers...")

    # Create both servers
    httpd_http, http_protocol = create_server(http_port, False)
    httpd_https, https_protocol = create_server(https_port, True, certfile, keyfile)

    # Run both servers in parallel using threads
    def run_http():
        logger.info(f"{ICONS['http']} HTTP server thread started on port {http_port}")
        try:
            httpd_http.serve_forever()
        except Exception as e:
            logger.error(f"{ICONS['error']} HTTP server error: {e}")

    def run_https():
        logger.info(f"{ICONS['https']} HTTPS server thread started on port {https_port}")
        try:
            httpd_https.serve_forever()
        except Exception as e:
            logger.error(f"{ICONS['error']} HTTPS server error: {e}")

    # Start HTTP server in main thread
    http_thread = threading.Thread(target=run_http, name="HTTP-Server")
    http_thread.daemon = False  # Wait for this thread on shutdown

    # Start HTTPS server in separate thread
    https_thread = threading.Thread(target=run_https, name="HTTPS-Server")
    https_thread.daemon = True  # Daemon thread for HTTPS

    logger.info(f"{ICONS['thread']} Starting server threads...")
    https_thread.start()
    http_thread.start()

    logger.info(f"{ICONS['success']} Both servers are running!")
    logger.info(f"  {ICONS['http']} HTTP:  http://0.0.0.0:{http_port}")
    logger.info(f"  {ICONS['https']} HTTPS: https://0.0.0.0:{https_port}")

    # Keep main thread alive for HTTP server
    try:
        http_thread.join()
    except KeyboardInterrupt:
        logger.info(f"{ICONS['server_stop']} Shutting down servers...")
        httpd_http.shutdown()
        httpd_https.shutdown()
        logger.info(f"{ICONS['success']} Servers stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple HTTP/HTTPS server that displays request headers and path")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--https", action="store_true", help="Use HTTPS instead of HTTP")
    parser.add_argument("--https-port", type=int, default=8443, help="HTTPS port to listen on (default: 8443)")
    parser.add_argument("--certfile", help="Path to SSL certificate file")
    parser.add_argument("--keyfile", help="Path to SSL private key file")
    parser.add_argument("--both", action="store_true", help="Run both HTTP and HTTPS servers")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--once", action="store_true", help="Handle one request and exit")

    args = parser.parse_args()

    # Adjust logging level if debug is enabled
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info(f"{ICONS['debug']} Debug logging enabled")

    try:
        if args.both:
            # Run both HTTP and HTTPS servers
            run_servers(args.port, args.https_port, args.certfile, args.keyfile)
        elif args.https:
            # Run HTTPS server only
            httpd, protocol = create_server(args.port, True, args.certfile, args.keyfile, args.once)
            logger.info(f"{ICONS['server_start']} HTTPS server running...")
            httpd.serve_forever()
        elif args.once:
            # Run HTTP server once and exit after one request
            httpd, protocol = create_server(args.port, False, None, None, True)
            logger.info(f"{ICONS['server_start']} HTTP server running (will exit after one request)...")
            httpd.serve_forever()
        else:
            # Run HTTP server only
            httpd, protocol = create_server(args.port, False, None, None, False)
            logger.info(f"{ICONS['server_start']} HTTP server running...")
            httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info(f"\n{ICONS['server_stop']} Server stopped by user")
    except Exception as e:
        logger.error(f"{ICONS['error']} Server error: {e}")
        sys.exit(1)
