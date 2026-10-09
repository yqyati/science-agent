"""Read-only network diagnostics executed inside the Sandbox."""

import json
import os
import re
import shutil
import socket
import ssl
import struct
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit


def diagnose_tls(host):
    """Inspect public peer certificates without sending HTTP or credentials."""
    if not shutil.which('openssl'):
        print('TLS chain', host, ': openssl unavailable')
        return
    command = ['openssl', 's_client', '-connect', host + ':443',
               '-servername', host, '-showcerts', '-verify_hostname', host]
    ca_file = ssl.get_default_verify_paths().cafile
    if ca_file:
        command += ['-CAfile', ca_file]
    try:
        result = subprocess.run(command, input='', capture_output=True, text=True, timeout=12)
    except subprocess.TimeoutExpired:
        print('TLS chain', host, ': timeout after 12s')
        return
    except OSError as exc:
        print('TLS chain', host, ':', str(exc))
        return
    print('TLS chain', host, ':')
    output = result.stdout + '\n' + result.stderr
    for line in output.splitlines():
        if any(marker in line for marker in ('subject=', 'issuer=', 'Verify return code:', 'Verification error:', 'verify error:', 'depth=')):
            print(line.strip())
    certificates = re.findall(r'-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----', result.stdout, re.S)
    for index, certificate in enumerate(certificates):
        info = subprocess.run(['openssl', 'x509', '-noout', '-subject', '-issuer', '-dates', '-fingerprint', '-sha256'],
                              input=certificate, capture_output=True, text=True, timeout=3)
        print('certificate', index, ':', info.stdout.strip().replace('\n', '; '))


def diagnose():
    print('TLS CA file:', ssl.get_default_verify_paths().cafile)
    try:
        print('TLS trusted CA count:', len(ssl.create_default_context().get_ca_certs()))
    except Exception as exc:
        print('TLS CA error:', type(exc).__name__, str(exc))
    resolver = Path('/etc/resolv.conf').read_text()
    print('resolv.conf:\n' + resolver)
    proxy_hosts = set()
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        value = os.getenv(name)
        if value:
            parsed = urlsplit(value if '://' in value else 'http://' + value)
            # Never print proxy credentials or unrelated environment variables.
            print(name + ' host:', parsed.hostname, 'port:', parsed.port)
            if parsed.hostname:
                proxy_hosts.add(parsed.hostname)
    print('proxy configured:', bool(proxy_hosts))
    targets = {'example.com', *proxy_hosts}
    api_host = urlsplit(os.getenv('API_BASE', '')).hostname
    if api_host:
        targets.add(api_host)
    code = 'import socket,sys,json; print(json.dumps(sorted({x[4][0] for x in socket.getaddrinfo(sys.argv[1],443)})))'
    for host in sorted(targets):
        try:
            result = subprocess.run([sys.executable, '-c', code, host], capture_output=True, text=True, timeout=8)
            print('resolve', host, ':', (result.stdout or result.stderr).strip()[-500:])
        except subprocess.TimeoutExpired:
            print('resolve', host, ': timeout after 8s')
    addresses = [line.split()[1] for line in resolver.splitlines() if line.strip().startswith('nameserver ')]
    # Query configured resolvers directly to distinguish DNS routing from libc.
    qname = b''.join(bytes([len(label)]) + label.encode() for label in 'example.com'.split('.')) + b'\0'
    query = struct.pack('!6H', 12345, 0x0100, 1, 0, 0, 0) + qname + struct.pack('!2H', 1, 1)
    for address in addresses[:3]:
        family = socket.AF_INET6 if ':' in address else socket.AF_INET
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as sock:
                sock.settimeout(3)
                sock.sendto(query, (address, 53))
                data, _ = sock.recvfrom(4096)
            header = struct.unpack('!6H', data[:12])
            print('DNS UDP', address, ': rcode=', header[1] & 15, 'answers=', header[3])
        except Exception as exc:
            print('DNS UDP', address, ':', type(exc).__name__, str(exc))
    endpoints = [('1.1.1.1', 443)]
    if api_host:
        try:
            socket.inet_aton(api_host)
            parsed = urlsplit(os.environ['API_BASE'])
            endpoints.append((api_host, parsed.port or (443 if parsed.scheme == 'https' else 80)))
        except OSError:
            pass
    for host, port in endpoints:
        try:
            with socket.create_connection((host, port), timeout=3):
                print('direct TCP', host, port, ': connected')
        except Exception as exc:
            print('direct TCP', host, port, ':', type(exc).__name__, str(exc))
    for host in ('example.com', 'google.serper.dev', 'r.jina.ai'):
        diagnose_tls(host)


if __name__ == '__main__':
    diagnose()
