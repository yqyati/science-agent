"""Initialize Ray with DSH settings, then execute the unmodified Molt CLI."""
import os
import runpy

import ray
from molt.cli import train_rl_ray


def install_rollout_storage_fallback():
    """Apply only to rollout dumps inside this recipe's Ray workers."""
    import logging
    import time
    import uuid
    from pathlib import Path
    import torch
    from molt.utils.io import is_storage_full

    original_save = torch.save
    if getattr(original_save, "_dsh_storage_fallback", False):
        return

    def save(obj, f, *args, **kwargs):
        if not isinstance(f, (str, os.PathLike)) or not Path(f).name.startswith("rollout_step"):
            return original_save(obj, f, *args, **kwargs)
        path = Path(f).absolute()
        destinations = [path] + [Path(root) / str(path).lstrip("/")
                               for root in os.environ.get("MOLT_CKPT_FALLBACK_ROOT", "").split(os.pathsep) if root]
        while True:
            for destination in dict.fromkeys(destinations):
                temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
                try:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    # A Python file preserves ENOSPC/EDQUOT from the writer.
                    with temporary.open("wb") as stream:
                        original_save(obj, stream, *args, **kwargs)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, destination)
                    logging.getLogger(__name__).info("DSH rollout saved: %s", destination)
                    return
                except Exception as exc:
                    if not is_storage_full(exc):
                        raise
                    logging.getLogger(__name__).warning("Storage full at %s; trying next destination", destination)
                finally:
                    temporary.unlink(missing_ok=True)
            logging.getLogger(__name__).error("All rollout destinations full; retrying in 30s")
            time.sleep(30)

    save._dsh_storage_fallback = True
    torch.save = save


if __name__ == "__main__":
    env = train_rl_ray._ray_runtime_env_vars()
    for key, value in os.environ.items():
        if key.startswith(("DSH_", "MOLT_")) or key in {
            "PYTHONPATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
            "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy",
        }:
            env[key] = value
    ray.init(address=os.environ.get("RAY_ADDRESS"), runtime_env={
        "env_vars": env, "worker_process_setup_hook": install_rollout_storage_fallback})
    runpy.run_path(train_rl_ray.__file__, run_name="__main__")
