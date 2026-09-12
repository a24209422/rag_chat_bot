# onperm/llm.py（llama.cpp server 的生成，走 OpenAI 相容 API）
import json

import requests

from shared.llm import BaseLLM, Reply, Stream, Usage
from shared.settings import settings


class LlamaCppLLM(BaseLLM):
    def __init__(self, url=None, timeout=None, config=None):
        cfg = config or settings()
        self.url = url or cfg.llama_url
        self.timeout = timeout if timeout is not None else cfg.llama_timeout
        self.model = "llama.cpp"          # server 載哪個模型由啟動參數決定，這裡不管

    def _payload(self, messages, system, temperature):
        # OpenAI 格式把 system 當成 messages 的第一則。它不進 history，每次現加。
        return {"messages": [{"role": "system", "content": system}, *messages],
                "temperature": temperature}

    def complete(self, messages, system, temperature=0.2):
        resp = requests.post(self.url, json=self._payload(messages, system, temperature),
                             timeout=self.timeout)
        resp.raise_for_status()

        data = resp.json()
        if "choices" not in data:
            raise RuntimeError(f"server 回了非預期內容：{data}")
        return Reply(data["choices"][0]["message"]["content"], self._usage(data))

    def stream(self, messages, system, temperature=0.2):
        payload = self._payload(messages, system, temperature)
        payload["stream"] = True
        # ⚠ 沒有這行就拿不到 usage。實測 llama.cpp 在 stream=True 時預設不回
        #   usage 欄位，要明確要求才給——跟非串流那條路不一樣。
        payload["stream_options"] = {"include_usage": True}

        resp = requests.post(self.url, json=payload, timeout=self.timeout, stream=True)
        resp.raise_for_status()
        # ⚠ 也是必要的。SSE 的 Content-Type 是 text/event-stream 而且沒宣告
        #   charset，requests 依 HTTP 對 text/* 的規定退回 ISO-8859-1——
        #   中文會變成 'å¥½ç\x9a\x84' 這種東西，而且不會報錯。
        resp.encoding = "utf-8"

        return Stream(self._parse_sse(resp))

    @staticmethod
    def _parse_sse(resp):
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue                       # 空行是事件分隔，註解行以 ':' 開頭
            body = line[6:]
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            piece = None
            for choice in chunk.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
            usage = chunk.get("usage")
            yield piece, (LlamaCppLLM._usage(chunk) if usage else None)

    def explain(self, exc):
        if isinstance(exc, requests.exceptions.ConnectionError):
            return f"連不上 {self.url}——llama.cpp server 沒開。"
        if isinstance(exc, requests.exceptions.Timeout):
            return f"等超過 {self.timeout} 秒，模型可能還在載入，稍等再問。"
        return None

    @staticmethod
    def _usage(data):
        """llama.cpp 實測會回 usage，但用 .get() 取——不同 build 或別的
        OpenAI 相容 server 不一定有，少了就當 0，不要讓整次對話失敗。"""
        u = data.get("usage") or {}
        return Usage(prompt=u.get("prompt_tokens") or 0,
                     output=u.get("completion_tokens") or 0)
