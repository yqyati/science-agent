"""Live smoke test: bash run_sandbox_harness.sh python3 sandbox_smoke.py."""

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "inference"))

from sandbox_gateway import close_shared_sandbox_run, shared_sandbox_run
from tool_http_client import tool_http_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true", help="Also call the summary LLM from the Sandbox")
    parser.add_argument("--page-file", action="store_true", help="Save a page in /workspace and read evidence with Python (requires qwen-agent)")
    parser.add_argument("--judge", action="store_true", help="Test the configured evaluation judge through Sandbox")
    parser.add_argument("--tools", action="store_true", help="Also run the real Search and Visit tools")
    parser.add_argument("--vision", action="store_true", help="Test input-image upload, crop/display, and matplotlib output")
    parser.add_argument("--search", action="store_true", help="Test Serper web and scholar search through Sandbox")
    parser.add_argument("--diagnose-network", action="store_true", help="Inspect Sandbox DNS/proxy/IP connectivity without calling external tools")
    parser.add_argument("--http-url", default="https://r.jina.ai/https://example.com", help="HTTPS URL to test; defaults to the Jina page-reading route")
    args = parser.parse_args()
    if args.search and not (os.getenv("SERPER_API_KEY") or os.getenv("SERPER_KEY_ID")):
        parser.error("Fill SERPER_API_KEY in sandbox.local.env for web and scholar search")
    if os.getenv("SCIENCE_AGENT_TOOL_HTTP_BACKEND") != "gateway_sandbox":
        parser.error("Launch with bash run_sandbox_harness.sh python3 sandbox_smoke.py")
    if args.llm or args.tools:
        missing = [name for name in ("API_BASE", "API_KEY", "SUMMARY_MODEL_NAME") if not os.getenv(name)]
        if missing:
            parser.error("Fill sandbox.local.env: " + ", ".join(missing))
    if args.judge:
        missing = [name for name in ("SCIENCE_AGENT_JUDGE_BASE_URL", "SCIENCE_AGENT_JUDGE_MODEL") if not os.getenv(name)]
        if not (os.getenv("SCIENCE_AGENT_JUDGE_API_KEY") or os.getenv("API_KEY")):
            missing.append("SCIENCE_AGENT_JUDGE_API_KEY or API_KEY")
        if missing:
            parser.error("Fill sandbox.local.env: " + ", ".join(missing))
    try:
        run = shared_sandbox_run()
        print("sandbox_id:", run.ensure(), flush=True)
        print("requested network default action:", os.getenv("SCIENCE_AGENT_SANDBOX_NETWORK_DEFAULT_ACTION") or "platform default", flush=True)
        assert run.run_python("x = 40\nprint(x + 2)").strip() == "42"
        assert run.run_python("print(x + 1)").strip() == "41"
        print("python persistence: passed", flush=True)
        if args.vision:
            from PIL import Image
            image = Image.new("RGB", (32, 24), "red")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            files = {"input_images/smoke.png": base64.b64encode(buffer.getvalue()).decode("ascii")}
            data = run.run_visual(
                "from PIL import Image\n"
                "img = Image.open('input_images/smoke.png')\n"
                "assert img.size == (32, 24)\n"
                "display(img.crop((0,0,8,8)))\n"
                "import matplotlib.pyplot as plt\n"
                "plt.plot([0,1],[0,1])\nplt.show()\n"
                "print('visual_measurement_summary: crop 8x8; shared x=', x)", files,
            )
            assert not data.get("error"), data.get("error")
            assert len(data.get("images", [])) == 2, "Expected crop and matplotlib images"
            crop = Image.open(io.BytesIO(base64.b64decode(data["images"][0]["data"])))
            assert crop.size == (8, 8)
            assert "shared x= 40" in data["stdout"]
            assert run.run_python("print(img.size)").strip() == "(32, 24)"
            print("vision upload/crop/plot/shared Python state: passed", flush=True)
        if args.diagnose_network:
            print_network_diagnostics(run)
            return
        print("outbound HTTP target:", args.http_url, flush=True)
        reader_headers = {}
        from urllib.parse import urlsplit
        if urlsplit(args.http_url).hostname == "r.jina.ai":
            keys = os.getenv("JINA_API_KEYS") or os.getenv("JINA_API_KEY", "")
            key = next((value.strip() for value in keys.split(",") if value.strip()), "")
            if key:
                reader_headers["Authorization"] = "Bearer " + key
        response = tool_http_request(
            prefix="JINA", method="GET", url=args.http_url, headers=reader_headers,
            trace_layer="sandbox_smoke", trace_event="sandbox_http",
        )
        print("outbound HTTP:", response.status_code, response.backend, flush=True)
        if response.status_code >= 400:
            print("outbound HTTP error:", response.text[:1500], flush=True)
            if response.status_code == 599:
                print_network_diagnostics(run)
        response.raise_for_status()
        assert response.backend == "gateway_sandbox"
        if args.http_url == "https://r.jina.ai/https://example.com":
            assert "Example Domain" in response.text, "Jina returned no expected page content"
        else:
            assert response.text, "HTTP response was empty"
        if args.search:
            for label, endpoint, query in (
                ("web", os.getenv("SERPER_API_URL", "https://google.serper.dev/search"), "Example Domain IANA"),
                ("scholar", os.getenv("SERPER_SCHOLAR_API_URL", "https://google.serper.dev/scholar"), "Gaussian basis sets 6-311G"),
            ):
                response = tool_http_request(
                    prefix="SERPER", method="POST", url=endpoint,
                    headers={"X-API-KEY": os.getenv("SERPER_API_KEY") or os.environ["SERPER_KEY_ID"]},
                    json_payload={"q": query}, trace_layer="sandbox_smoke", trace_event="sandbox_search",
                )
                print("Serper", label, "HTTP:", response.status_code, flush=True)
                if response.status_code >= 400:
                    print("Serper error:", response.text[:1500], flush=True)
                response.raise_for_status()
                assert response.json().get("organic"), "Serper returned no search results"
                print("Serper", label, "via Sandbox: passed", flush=True)
        if args.llm:
            response = tool_http_request(
                prefix="SUMMARY", method="POST",
                url=os.environ["API_BASE"].rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + os.environ["API_KEY"]},
                json_payload={"model": os.environ["SUMMARY_MODEL_NAME"], "messages": [{"role": "user", "content": "Reply with OK."}]},
                trace_layer="sandbox_smoke", trace_event="sandbox_llm",
            )
            response.raise_for_status()
            assert response.json()["choices"][0]["message"].get("content")
            print("summary LLM via Sandbox: passed", flush=True)
        if args.page_file:
            from tool_visit_mcp_r1 import Visit
            tool = Visit()
            url = "http://example.com"
            metadata = json.loads(tool.call({"url": url, "goal": "Find the page title"}))
            assert "absolute_path" not in metadata
            assert metadata["path"].startswith("pages/") and not Path(metadata["path"]).is_absolute()
            path = metadata["path"]
            output = run.run_python("import os\nfrom pathlib import Path\nassert os.getcwd() == '/workspace'\n" +
                                    "text = Path(" + repr(path) + ").read_text(encoding='utf-8')\n" +
                                    "assert len(text) == " + str(metadata["characters"]) + "\n" +
                                    "print('saved page evidence:', text[:300])")
            assert "saved page evidence:" in output, output
            repeated = json.loads(tool.call({"url": url, "goal": "Read again"}))
            assert repeated["cached"] and repeated["path"] == path
            from tool_python import PythonInterpreter
            python_tool = PythonInterpreter()
            evidence = python_tool.call({"code": "from pathlib import Path\nprint(Path(" + repr(path) + ").read_text(encoding='utf-8')[:300])"})
            assert evidence.strip(), evidence
            large = python_tool.call({"code": "print('x' * 20000)"})
            assert len(large) <= int(os.getenv("PYTHON_TOOL_RETURN_MAX_CHARS", "6000")), len(large)
            assert "output truncated" in large, large[:100]
            print("page saved/Python tool read/cache/output limit: passed", flush=True)
        if args.judge:
            response = run.run_http({
                "method": "POST", "url": os.environ["SCIENCE_AGENT_JUDGE_BASE_URL"].rstrip("/") + "/chat/completions",
                "headers": {"Authorization": "Bearer " + (os.getenv("SCIENCE_AGENT_JUDGE_API_KEY") or os.environ["API_KEY"])},
                "json": {"model": os.environ["SCIENCE_AGENT_JUDGE_MODEL"], "temperature": 0.0,
                         "max_tokens": int(os.getenv("SCIENCE_AGENT_JUDGE_MAX_TOKENS", "1800")),
                         "response_format": {"type": "json_object"},
                         "messages": [{"role": "user", "content": 'Grade candidate 2 against reference 2 for 1+1. Return JSON {"score": 1, "correct": true}.'}]},
            }, timeout=float(os.getenv("SCIENCE_AGENT_JUDGE_TIMEOUT", "180")))
            assert response["status_code"] == 200, f"Judge HTTP {response['status_code']}: {response['text'][:500]}"
            grade = json.loads(json.loads(response["text"])["choices"][0]["message"]["content"])
            assert grade["score"] == 1 and grade["correct"] is True, grade
            print("judge", os.environ["SCIENCE_AGENT_JUDGE_MODEL"], "via Sandbox: passed", flush=True)
        if args.tools:
            from tool_search_mcp_r1 import Search
            from tool_visit_mcp_r1 import Visit
            print("search:", Search().call({"query": "Example Domain IANA"}), flush=True)
            print("visit:", Visit().call({"url": "https://example.com", "goal": "Explain the purpose of this domain."}), flush=True)
        assert shared_sandbox_run() is run
        print("shared run: passed", flush=True)
    finally:
        close_shared_sandbox_run()
        print("cleanup: done", flush=True)


def print_network_diagnostics(run):
    print("Sandbox network diagnostics:", flush=True)
    source = Path(__file__).resolve().parent / "inference/sandbox_network_diagnostics.py"
    code = "exec(compile(" + repr(source.read_text()) + ", '<network diagnostics>', 'exec'), {'__name__': '__main__'})"
    try:
        print(run.run_python(code, timeout=90), flush=True)
    except Exception as exc:
        print("network diagnostics failed:", type(exc).__name__, str(exc), flush=True)


if __name__ == "__main__":
    main()
