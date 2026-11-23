"""Dependency and real filesystem-isolation check; makes no model API calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from isolation import ROOT, command


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "vendor"))
    from deepseek_harness import DeepSeekHarness
    from deepseek_harness_runtime import resolve_bundled_launch_args
    pipeline = Path(os.environ["DSH_VERIFIER_ROOT"])
    entry = "verify_candidate.py" if (pipeline / "verify_candidate.py").is_file() else "tools/verify_task.py"
    if not (pipeline / entry).is_file():
        raise FileNotFoundError(pipeline / entry)
    executable = resolve_bundled_launch_args("exe")[0]
    with tempfile.TemporaryDirectory(prefix="dsh-isolation-") as tmp:
        tmp = Path(tmp)
        workspace, home = tmp / "workspace", tmp / "home"
        workspace.mkdir()
        home.mkdir()
        (tmp / "private-verifier").write_text("must remain hidden")
        env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "NO_PROXY": "*"}
        args = command(["/usr/bin/python3", "-c",
                        "from pathlib import Path; "
                        f"assert not Path({str(tmp / 'private-verifier')!r}).exists(); "
                        "assert not Path('/proc/1/root').exists(); "
                        "Path('isolation-ok').write_text('ok')"], workspace, home, env)
        if os.geteuid() == 0:
            for path in [tmp, workspace, home]:
                os.chown(path, 65534, 65534)
        process = subprocess.run(args, text=True, capture_output=True, timeout=20,
                                 user=65534 if os.geteuid() == 0 else None,
                                 group=65534 if os.geteuid() == 0 else None)
        if process.returncode:
            raise RuntimeError("Bubblewrap isolation unavailable: " + process.stderr[-2000:])
    print(json.dumps({"sdk": "ok", "runtime": executable, "filesystem_isolation": "ok",
                      "verifier_root": str(pipeline), "gpu_training": "not tested"}))
