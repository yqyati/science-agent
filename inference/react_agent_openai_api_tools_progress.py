"""Progress-verifier variant of the OpenAI tools agent; original stays intact."""

from react_agent_openai_api_tools import MultiTurnReactAgent as BaseAgent
from react_agent_qwen_api_progress import ProgressVerifierMixin


class MultiTurnReactAgent(ProgressVerifierMixin, BaseAgent):
    pass
