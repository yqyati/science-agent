# Legacy ReAct harness snapshot

This directory is a fixed vendor snapshot used by the Molt science Legacy
ReAct adapter. It was copied on 2026-09-30 from:

`/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/liuzhuo/science-agent`

Only the local Python import closure of
`inference/react_agent_openai_api_tools.py` is included (19 source files).
Caches, run outputs, datasets, and the source checkout's mutable state are
intentionally excluded. Future training jobs should use this directory through
`SCIENCE_AGENT_ROOT`; setting that variable remains an explicit override.

Core-file SHA256 at snapshot creation:

```text
6890a4f916efc5de6c0717c61eaf6e7a99131904a419def4f4c60017d2798d1b  inference/react_agent_openai_api_tools.py
7b193bc6252207d56cfff8634eddc8b34d433767ed1d3298f2983a8ec5069e8e  inference/react_agent_qwen_api.py
56053e7b9142232860a688b664eadab9a3463069db1fadea047367fb46053f29  inference/tool_runtime_registry.py
f7308de984177080a19230c0af2fba45699d5d20b641c75b700de170d9873c90  science_agent/tracing.py
55642f40084c113481b1158eb870e1a7abb589bdeb168ab6a2060c3437bf345b  science_agent/tool_call_parsing.py
```
