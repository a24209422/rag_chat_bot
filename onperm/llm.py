# onperm/llm.py（llama.cpp server 的生成，走 OpenAI 相容 API）
import requests

from shared.llm import BaseLLM, Reply, Usage
from shared.settings import settings


class LlamaCppLLM(BaseLLM):
    def __init__(self, url=None, timeout=None, config=None):
        cfg = config or settings()
        self.url = url or cfg.llama_url
        self.timeout = timeout if timeout is not None else cfg.llama_timeout
        self.model = "llama.cpp"          # server 載哪個模型由啟動參數決定，這裡不管

    def complete(self, messages, system, temperature=0.2):
        # OpenAI 格式把 system 當成 messages 的第一則。它不進 history，每次現加。
        payload = {"messages": [{"role": "system", "content": system}, *messages],
                   "temperature": temperature}
        resp = requests.post(self.url, json=payload, timeout=self.timeout)
        resp.raise_for_status()

        data = resp.json()
        if "choices" not in data:
            raise RuntimeError(f"server 回了非預期內容：{data}")
        return Reply(data["choices"][0]["message"]["content"], self._usage(data))

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
