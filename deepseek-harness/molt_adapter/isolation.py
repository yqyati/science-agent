"""A private filesystem per DSH process; verifier/data/other episodes stay outside."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def command(argv, workspace, home, env):
    bwrap = ROOT / "vendor/bwrap"
    args = [str(bwrap), "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
            "--die-with-parent", "--new-session", "--cap-drop", "ALL"]
    for name in ("/usr", "/bin", "/sbin", "/lib", "/lib64"):
        path = Path(name)
        if path.is_symlink():
            args += ["--symlink", os.readlink(path), name]
        elif path.exists():
            args += ["--ro-bind", name, name]
    for name in ("/etc/alternatives", "/etc/ssl", "/etc/resolv.conf", "/etc/hosts",
                 "/etc/nsswitch.conf", "/etc/passwd", "/etc/group", "/etc/ld.so.cache"):
        if Path(name).exists():
            args += ["--ro-bind", name, name]
    args += ["--dir", "/proc", "--dir", "/proc/self", "--symlink", argv[0], "/proc/self/exe",
             "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/var/tmp", "--tmpfs", "/dev/shm",
             "--ro-bind", str(ROOT / "vendor"), str(ROOT / "vendor"),
             "--bind", str(workspace), str(workspace), "--bind", str(home), str(home), "--clearenv"]
    for key, value in env.items():
        args += ["--setenv", key, str(value)]
    return args + ["--chdir", str(workspace), "--", *map(str, argv)]


if __name__ == "__main__":
    config = json.loads(sys.argv[1])
    # The SDK injects only this per-episode policy endpoint into the launcher.
    env = config["env"]
    for key in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"):
        if key in os.environ:
            env[key] = os.environ[key]
    args = command(config["argv"], Path(config["workspace"]), Path(config["home"]), env)
    os.execv(args[0], args)
