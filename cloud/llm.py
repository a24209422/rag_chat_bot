# cloud/llm.py（Gemini 的生成）
from google.genai import types

from cloud.client import make_client
from shared.llm import BaseLLM, Reply, Stream, Usage
from shared.settings import settings


class GeminiLLM(BaseLLM):
    def __init__(self, client=None, model=None, config=None):
        cfg = config or settings()
        self._client = client
        self.model = model or cfg.gemini_chat_model
        self._cfg = cfg

    @property
    def client(self):
        if self._client is None:                 # 延遲：建構不讀金鑰
            self._client = make_client(config=self._cfg)
        return self._client

    def complete(self, messages, system, temperature=0.2):
        resp = self.client.models.generate_content(
            model=self.model,
            contents=[self._wire(m) for m in messages],
            config=types.GenerateContentConfig(
                system_instruction=system,        # ← Gemini 有獨立欄位，不塞進 contents
                temperature=temperature,
            ),
        )
        text = resp.text
        if not text:
            raise RuntimeError("沒拿到內容")
        return Reply(text, self._usage(resp.usage_metadata))

    def explain(self, exc):
        from google.genai import errors
        if not isinstance(exc, errors.APIError):
            return None
        if exc.code == 503:
            return "Gemini 暫時過載，請稍後再試。"
        if exc.code == 429:
            # 免費方案觀測到的上限：這個模型每天 20 次生成。也可能是每分鐘超速
            return "已達 Gemini 免費方案的用量上限，請稍後或明天再試。"
        return None

    @staticmethod
    def _wire(msg):
        """中性格式 → Gemini 的線路格式。assistant 在 Gemini 叫 model。"""
        role = "user" if msg["role"] == "user" else "model"
        return {"role": role, "parts": [{"text": msg["content"]}]}

    @staticmethod
    def _usage(u):
        if u is None:
            return Usage()
        return Usage(
            prompt=u.prompt_token_count or 0,     # 每個欄位都是 Optional[int]
            # thoughts 是使用者看不到的內部草稿，但按輸出計費——不加會嚴重低估
            output=(u.candidates_token_count or 0) + (u.thoughts_token_count or 0),
        )

    def stream(self, messages, system, temperature=0.2):
        def pieces():
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=[self._wire(m) for m in messages],
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                ),
            ):
                u = chunk.usage_metadata
                # usage 取最後一個有值的。Gemini 可能每個 chunk 都帶（累計值）、
                # 也可能只有最後一個帶——兩種情況這樣寫都對。
                yield chunk.text, (self._usage(u) if u is not None else None)

        return Stream(pieces())
