"""Configure a verified DNS fallback inside one Sandbox instance."""

import json
import socket
import struct
import sys
from pathlib import Path


def resolver_works(address):
    qname = b''.join(bytes([len(label)]) + label.encode() for label in 'example.com'.split('.')) + b'\0'
    query = struct.pack('!6H', 19423, 0x0100, 1, 0, 0, 0) + qname + struct.pack('!2H', 1, 1)
    family = socket.AF_INET6 if ':' in address else socket.AF_INET
    with socket.socket(family, socket.SOCK_DGRAM) as sock:
        sock.settimeout(3)
        sock.connect((address, 53))
        sock.send(query)
        data = sock.recv(4096)
    ident, flags, questions, answers, _, _ = struct.unpack('!6H', data[:12])
    return ident == 19423 and bool(flags & 0x8000) and flags & 15 == 0 and answers > 0


def configure(candidates):
    path = Path('/etc/resolv.conf')
    original = path.read_text()
    errors = []
    working = []
    for address in candidates:
        try:
            if resolver_works(address):
                working.append(address)
            else:
                errors.append(address + ': no successful DNS answer')
        except Exception as exc:
            errors.append(address + ': ' + type(exc).__name__ + ': ' + str(exc))
    if not working:
        raise RuntimeError('No fallback DNS responded: ' + '; '.join(errors))
    # Keep Kubernetes search domains, but use one dot to avoid repeated search
    # expansion for public domains. Only this Sandbox's resolver is modified.
    retained = [line for line in original.splitlines() if not line.strip().startswith(('nameserver ', 'options '))]
    updated = '\n'.join(retained + ['nameserver ' + address for address in working] + ['options ndots:1 timeout:2 attempts:2']) + '\n'
    try:
        with path.open('w') as handle:
            handle.write(updated)
        resolved = sorted({entry[4][0] for entry in socket.getaddrinfo('example.com', 443)})
    except Exception:
        with path.open('w') as handle:
            handle.write(original)
        raise
    return {'dns_servers': working, 'resolved': resolved}


if __name__ == '__main__':
    try:
        print(json.dumps(configure(sys.argv[1:]), ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({'error': type(exc).__name__ + ': ' + str(exc)}, ensure_ascii=False))
        sys.exit(1)
