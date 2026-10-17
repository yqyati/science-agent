"""Real local worker and gateway timeout lifecycle; no external services."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
import uuid
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
import sandbox_gateway as gateway

WORKER = Path(gateway.__file__).with_name('sandbox_python_worker.py')


class PythonTimeoutProcessTests(unittest.TestCase):
    def test_interrupt_then_restart_preserves_files_and_clears_code_history(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            'SCIENCE_AGENT_SANDBOX_RECOVERY': '1',
            'SCIENCE_AGENT_SANDBOX_RECOVERY_DIR': directory + '/journal',
            'SCIENCE_AGENT_SANDBOX_PYTHON_WORKER': str(WORKER),
        }):
            run = gateway.AgentSandboxRun()
            run.sandbox_id = 'local-test'
            processes = []
            identities = []
            address = ['']

            class Client:
                _shell_quote = staticmethod(gateway.SandboxGatewayClient._shell_quote)
                command_output = staticmethod(gateway.SandboxGatewayClient.command_output)

                def proxy_json(self, sid, port, path, payload, timeout=10):
                    req = urllib.request.Request(address[0] + path, data=json.dumps(payload).encode(),
                                                 headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(req, timeout=timeout) as response:
                        return json.load(response)

                def proxy_http(self, sid, port, path, timeout=10):
                    with urllib.request.urlopen(address[0] + path, timeout=timeout) as response:
                        return response.status, response.read().decode()

                def command(self, sid, command, timeout=15):
                    result = subprocess.run([sys.executable, str(WORKER), '--stop', run._python_worker_id],
                                            capture_output=True, text=True, timeout=timeout, check=True)
                    return json.dumps({'type': 'stdout', 'text': result.stdout})

            def start(kind, port):
                if run._python_started:
                    return run.sandbox_id
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0))
                    port = sock.getsockname()[1]
                identity = uuid.uuid4().hex
                identities.append(identity)
                run._python_worker_id = identity
                address[0] = f'http://127.0.0.1:{port}'
                process = subprocess.Popen([sys.executable, str(WORKER), '--identity', identity],
                    cwd=directory, env=dict(os.environ, SANDBOX_PYTHON_WORKER_PORT=str(port)),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                processes.append(process)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        if run.client.proxy_http('', port, '/identity', timeout=0.2)[1] == identity:
                            run._python_started = True
                            return run.sandbox_id
                    except OSError:
                        time.sleep(0.05)
                raise AssertionError('worker startup failed')

            run.client = Client()
            run._start_worker = start
            try:
                self.assertEqual(run.run_python('x = 40; print(x)', timeout=1), '40\n')
                identity = run._python_worker_id
                result = run.run_python('y = 2\nwhile True:\n    pass', timeout=0.3)
                self.assertIn('ExecutionInterrupted', result)
                self.assertEqual(run._python_worker_id, identity)
                self.assertIn('session was retained', run.take_recovery_notice())
                self.assertEqual(run.run_python('print(x+y)', timeout=1), '42\n')
                with self.assertRaisesRegex(gateway.SandboxStateError, 'session restarted'):
                    run.run_python("from pathlib import Path\nPath('kept.txt').write_text('kept')\nimport time\ntime.sleep(60)", timeout=0.3)
                self.assertNotEqual(run._python_worker_id, identity)
                self.assertIsNotNone(processes[0].poll())
                self.assertIn('Historical code was not replayed', run.take_recovery_notice())
                self.assertEqual(run.run_python("from pathlib import Path\nprint('x' in globals(), Path('kept.txt').read_text())", timeout=1), 'False kept\n')
                starts = [e for e in run._journal.events if e['kind'] == 'python_start']
                self.assertEqual(len(starts), 1)
                self.assertNotIn('sleep', starts[0]['payload']['code'])
            finally:
                for process in processes:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                for identity in identities:
                    Path(f'/tmp/agent-python-{identity}.json').unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
