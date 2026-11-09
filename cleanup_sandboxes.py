"""List or delete Sandbox instances recorded by one stopped run."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "inference"))
from sandbox_gateway import SandboxGatewayClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True, help="Exact RUN_NAME of the stopped run")
    parser.add_argument("--delete", action="store_true", help="Delete pending instances; default only lists them")
    args = parser.parse_args()
    if not args.ledger_dir.is_dir():
        parser.error(f"Ledger directory does not exist: {args.ledger_dir}")
    # Validate the whole directory before sending any deletion requests.
    pending = []
    for path in sorted(args.ledger_dir.glob("*.json")):
        record = json.loads(path.read_text())
        if record["run_name"] != args.run_name:
            parser.error(f"Run name mismatch in {path.name}; no instances were deleted")
        if record["status"] == "created":
            pending.append(record)
        elif record["status"] != "deleted":
            parser.error(f"Unknown status in {path.name}")
    token = os.getenv("OPEN_SANDBOX_GATEWAY_TOKEN", "")
    if args.delete and pending and not token:
        parser.error("Missing OPEN_SANDBOX_GATEWAY_TOKEN; launch through run_sandbox_harness.sh")
    failures = 0
    os.environ["SCIENCE_AGENT_SANDBOX_LEDGER_DIR"] = str(args.ledger_dir)
    for record in pending:
        sid = record["sandbox_id"]
        print(f"pending id={sid} resource={record['resource']}", flush=True)
        if not args.delete:
            continue
        client = SandboxGatewayClient(record["base_url"], token, record["resource"])
        for attempt in range(3):
            try:
                client.delete(sid)
                print(f"deleted id={sid}", flush=True)
                break
            except Exception as exc:
                # Keep credentials and response bodies out of cleanup output.
                print(f"delete failed id={sid} attempt={attempt + 1}/3 error={type(exc).__name__}", file=sys.stderr)
                if attempt == 2:
                    failures += 1
                else:
                    time.sleep(attempt + 1)
    print(f"pending={len(pending)} failed={failures} mode={'delete' if args.delete else 'list'}")
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(main())
