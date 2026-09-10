"""Bounded OpenAI-compatible judge calls using the standard library only.

Uses the project's existing GPT_4_URL/GPT_4_KEY and DEEPSEEK_API/DEEPSEEK_KEY.
Never prints credentials, request headers, or provider error bodies.
"""
from __future__ import annotations
import ast
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def settings() -> dict[str, str]:
    values = {}
    path = Path(__file__).resolve().parents[3] / '.env'
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.removeprefix('export ').split('=', 1)
            value = value.strip()
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                value = value.split(' #', 1)[0].strip()
            if isinstance(value, str):
                values[key.strip()] = value
    values.update(os.environ)
    return values


class JudgeClient:
    def __init__(self, model_name: str, timeout: int = 120, max_attempts: int = 2):
        self.model_name = model_name
        config = settings()
        deepseek = model_name.startswith('deepseek')
        self.url = config.get('DEEPSEEK_API' if deepseek else 'GPT_4_URL', '').rstrip('/')
        self.key = config.get('DEEPSEEK_KEY' if deepseek else 'GPT_4_KEY', '')
        if not self.url or not self.key:
            raise ValueError(f'Missing configured endpoint or credential for {model_name}')
        self.timeout, self.max_attempts = timeout, max_attempts
        self.calls = []

    def call(self, system: str, user: str) -> str:
        payload = {'model': self.model_name, 'messages': [
            {'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}
        if self.model_name.startswith('gpt-5'):
            payload.update(max_completion_tokens=2048, reasoning_effort='low')
        else:
            payload.update(max_tokens=1024, temperature=0)
        if self.model_name.startswith('deepseek'):
            payload['thinking'] = {'type': 'disabled'}
        for attempt in range(self.max_attempts):
            start = time.monotonic()
            request = urllib.request.Request(self.url + '/chat/completions',
                data=json.dumps(payload).encode(), headers={
                    'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
            event = {'requested_model': self.model_name, 'attempt': attempt + 1,
                     'generation_settings': {k: v for k, v in payload.items() if k not in {'messages', 'model'}}}
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.load(response)
                choice = data['choices'][0]
                content = choice['message'].get('content')
                event.update(returned_model=data.get('model'), usage=data.get('usage'),
                             finish_reason=choice.get('finish_reason'), status='ok')
                if not isinstance(content, str) or not content.strip() or choice.get('finish_reason') != 'stop':
                    raise ValueError('Empty or unfinished judge response')
                return content
            except urllib.error.HTTPError as exc:
                event.update(status='error', error_type='HTTPError', http_status=exc.code)
                if exc.code not in (429, 500, 502, 503, 504) or attempt + 1 == self.max_attempts:
                    raise RuntimeError(f'Judge HTTP {exc.code}: {self.model_name}') from None
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
                event.update(status='error', error_type=type(exc).__name__)
                if attempt + 1 == self.max_attempts:
                    raise RuntimeError(f'Judge request failed: {self.model_name} ({type(exc).__name__})') from None
            finally:
                event['elapsed_seconds'] = round(time.monotonic() - start, 3)
                self.calls.append(event)
            time.sleep(2 * (attempt + 1))
        raise RuntimeError('Judge attempts exhausted')
