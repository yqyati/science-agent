# Sandbox cleanup after a stopped run

When `SCIENCE_AGENT_TRACE_FILE` is configured, the gateway records each created
instance under `sandbox_ledger/` beside that trace file. Set
`SCIENCE_AGENT_SANDBOX_LEDGER_DIR` to override this location. Each JSON record
contains the instance ID, actual resource pool, gateway URL, RUN_NAME and status;
credentials are never stored. Successful deletion marks it `deleted`.

Only clean up after the run has stopped on **all nodes**. Run the independent
script from another session; it does not monitor RJob status automatically.
Use the exact RUN_NAME from the training output directory, not the RJob ID.

```bash
cd /mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/harness/science-agent-sandbox
python3 cleanup_sandboxes.py --ledger-dir /path/to/run/logs/sandbox_ledger --run-name RUN_NAME
bash run_sandbox_harness.sh python3 cleanup_sandboxes.py \
  --ledger-dir /path/to/run/logs/sandbox_ledger --run-name RUN_NAME --delete
```

The first command only lists pending instances. The second uses the current
Sandbox credential, retries failed deletions three times, and returns a nonzero
exit code if any remain. Repeating it skips instances already marked deleted;
a confirmed missing instance also counts as deleted. A mismatched RUN_NAME
stops cleanup before any deletion request.

Existing instances created before this change have no ledger. Unknown IDs from
a lost create response or a process killed before recording still depend on
the service's timeout policy. A ledger write failure after creation stops that
creation and attempts immediate deletion, rather than continuing untracked.
