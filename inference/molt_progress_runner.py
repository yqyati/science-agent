"""Evaluation-only same-weight verifier with independent transient sessions."""
from uuid import uuid4
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from molt.agents import ChatAgentRunner
from molt.agents._chat_server import (ChatServerState, _Session, _run_turn,
    _chat_completion_body, mount_session_capture, serve_on_current_loop)

class _VerifierProcessor:
    def __init__(self, processor):
        self._processor = processor
    def __getattr__(self, name):
        return getattr(self._processor, name)
    def __call__(self, *args, **kwargs):
        return self._processor(*args, **kwargs)
    def apply_chat_template(self, *args, **kwargs):
        kwargs['enable_thinking'] = False
        return self._processor.apply_chat_template(*args, **kwargs)

class ProgressChatAgentRunner(ChatAgentRunner):
    async def _ensure_server(self, llm_engine, hf_tokenizer, max_length, sampling_params):
        if self._state is not None:
            return
        async with self._boot_lock:
            if self._state is not None:
                return
            app = FastAPI()
            state = ChatServerState(llm_engine, hf_tokenizer, 'policy', max_length, sampling_params)
            verifier_state = ChatServerState(llm_engine, _VerifierProcessor(hf_tokenizer),
                                            'policy', max_length, sampling_params)
            mount_session_capture(app, state)
            @app.get('/health')
            async def health():
                return {'status': 'ok', 'model': 'policy', 'verifier': 'same_weights_independent_context'}
            @app.post('/verify/v1/chat/completions')
            async def verify(request: Request):
                body = await request.json()
                # Never register/stitch verifier tokens into the policy trajectory.
                session = _Session(prompt='', label=None, images=None,
                                   session_id='verify-' + uuid4().hex,
                                   sampling_params=sampling_params)
                try:
                    text, finish = await _run_turn(verifier_state, session, body)
                    return JSONResponse(_chat_completion_body('policy', text, finish))
                finally:
                    session.trajectories.clear()
            port, self._server_task = await serve_on_current_loop(app)
            self._server_root = f'http://127.0.0.1:{port}'
            self._state = state
