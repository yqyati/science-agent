"""Authenticate with the heterogeneous-platform AD gateway.

Run an authenticated command, or save the Token to a local environment file
for noninteractive jobs. AD passwords are never saved.
"""

from __future__ import annotations

import getpass
import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
import shlex
import tempfile


def login(base_url: str) -> str:
    username = os.getenv("AD_USERNAME") or input("AD 用户名: ").strip()
    password = getpass.getpass("AD 密码（不回显）: ")
    request = urllib.request.Request(
        base_url + "/api/v1/auth/ldap/login",
        data=json.dumps({"username": username, "password": password}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    finally:
        del password, request

    user = result.get("user") or {}
    if user.get("authSource") != "ldap" or user.get("status") != "active":
        raise RuntimeError("账号尚未审批授权，请联系管理员后重新运行。")
    token = result.get("token")
    if not token:
        raise RuntimeError("未取得个人 Token，请联系管理员。")

    request = urllib.request.Request(
        base_url + "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()
    return token


def save_token(path: str, token: str) -> None:
    target = Path(path).expanduser().resolve()
    lines = target.read_text().splitlines() if target.exists() else []
    lines = [line for line in lines if not line.strip().removeprefix("export ").startswith("OPEN_SANDBOX_GATEWAY_TOKEN=")]
    lines.append("OPEN_SANDBOX_GATEWAY_TOKEN=" + shlex.quote(token))
    descriptor, temporary = tempfile.mkstemp(prefix=".sandbox-env-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-token", metavar="ENV_FILE", help="Save Token to the local env file with mode 600")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command and not args.save_token:
        parser.error("指定 --save-token <配置文件> 或 -- <业务命令及参数>")

    base_url = os.getenv("SCIENCE_AGENT_SANDBOX_BASE_URL") or os.getenv("BASE_URL", "http://10.12.111.133:10018")
    base_url = base_url.rstrip("/")
    try:
        token = login(base_url)
    except urllib.error.HTTPError as exc:
        print(f"认证失败（HTTP {exc.code}）：请检查账号密码、审批状态或联系管理员。", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError, ValueError, RuntimeError) as exc:
        print(f"无法完成 AD 登录与授权检查：{exc}", file=sys.stderr)
        return 1

    if args.save_token:
        save_token(args.save_token, token)
        print("Sandbox Token 已保存到 " + str(Path(args.save_token).resolve()) + "（权限 600）；AD 密码未保存。")
    if not command:
        return 0
    env = os.environ.copy()
    env.update(
        BASE_URL=base_url,
        GATEWAY_TOKEN=token,
        OPEN_SANDBOX_GATEWAY_TOKEN=token,
        TOKEN=token,
    )
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
