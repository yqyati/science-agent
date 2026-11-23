"""Route the two independently reviewed target verifier packages."""
from agent_verifier import verify as agent_verify
from extra_verifier import verify as extra_verify

def verify(task, answer):
    if task["verifier_id"].startswith("target_extra:"):
        return extra_verify(task, answer)
    if task["verifier_id"].startswith("target400.verifier:"):
        return agent_verify(task, answer)
    raise ValueError("Unknown target verifier provenance")
