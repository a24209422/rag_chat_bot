"""各家 LLM client：中性格式怎麼轉成自己的線路格式、usage 怎麼讀。

這一層是 Stage 2 新增的。在它之前，這些轉換散在兩個 ChatBot 裡跟編排邏輯
混在一起，沒辦法單獨測；現在每家就是一個小類別，一組輸入對一組輸出。
"""
import pytest
import requests

from cloud.llm import GeminiLLM
from onperm.llm import LlamaCppLLM
from shared.llm import BaseLLM, Usage

NEUTRAL = [{"role": "user", "content": "第一題"},
           {"role": "assistant", "content": "第一答"},
           {"role": "user", "content": "第二題"}]


def test_base_沒實作complete():
    with pytest.raises(NotImplementedError):
        BaseLLM().complete([], system="s")


def test_base_的explain預設不認得任何錯誤():
    assert BaseLLM().explain(ValueError("x")) is None


def test_usage可以相加():
    """UI 用它累計整個 session。"""
    assert Usage(10, 5) + Usage(1, 2) == Usage(11, 7)


# ══ Gemini ════════════════════════════════════════════════════════════
class FakeUsageMeta:
    def __init__(self, prompt=100, candidates=50, thoughts=30):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.thoughts_token_count = thoughts


class FakeGeminiClient:
    def __init__(self, text="雲端的回答", usage=None):
        self._text, self._usage = text, usage if usage is not None else FakeUsageMeta()
        self.models = self
        self.seen = {}

    def generate_content(self, model=None, contents=None, config=None):
        self.seen = {"model": model, "contents": contents, "config": config}
        return type("R", (), {"text": self._text, "usage_metadata": self._usage})()


def test_gemini_把assistant轉成model():
    """Gemini 的線路格式用 model 而不是 assistant，內容包在 parts 裡。"""
    client = FakeGeminiClient()
    GeminiLLM(client=client).complete(NEUTRAL, system="系統指令")

    assert client.seen["contents"] == [
        {"role": "user", "parts": [{"text": "第一題"}]},
        {"role": "model", "parts": [{"text": "第一答"}]},
        {"role": "user", "parts": [{"text": "第二題"}]},
    ]


def test_gemini_的system走獨立欄位不進contents():
    client = FakeGeminiClient()
    GeminiLLM(client=client).complete(NEUTRAL, system="系統指令")

    assert client.seen["config"].system_instruction == "系統指令"
    assert all(c["role"] != "system" for c in client.seen["contents"])


def test_gemini_輸出token要含思考token():
    """thoughts 是使用者看不到的內部草稿，但按輸出計費——不加會嚴重低估。"""
    client = FakeGeminiClient(usage=FakeUsageMeta(100, 50, 30))
    reply = GeminiLLM(client=client).complete(NEUTRAL, system="s")

    assert reply.usage == Usage(prompt=100, output=80)      # 50 + 30


def test_gemini_的usage欄位可能是None():
    """google-genai 的每個 token 欄位都是 Optional[int]。"""
    client = FakeGeminiClient(usage=FakeUsageMeta(None, None, None))
    assert GeminiLLM(client=client).complete(NEUTRAL, system="s").usage == Usage(0, 0)


def test_gemini_沒拿到內容就丟例外():
    """要在這裡丟，ChatBot 才收得到並把 history 收回去。"""
    with pytest.raises(RuntimeError, match="沒拿到內容"):
        GeminiLLM(client=FakeGeminiClient(text="")).complete(NEUTRAL, system="s")


def test_gemini_建構不讀金鑰():
    """沒有 GEMINI_API_KEY 也要建得起來（測試與 CI 需要）。"""
    assert GeminiLLM()._client is None


@pytest.mark.parametrize("code, keyword", [(503, "過載"), (429, "用量上限")])
def test_gemini_翻譯已知錯誤(code, keyword):
    from google.genai import errors
    exc = errors.APIError(code, {"error": {"message": "x"}})
    assert keyword in GeminiLLM().explain(exc)


def test_gemini_不認得的錯誤回None():
    assert GeminiLLM().explain(ValueError("誰知道")) is None


# ══ llama.cpp ═════════════════════════════════════════════════════════
class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def ok_payload(text="地端的回答", usage=None):
    d = {"choices": [{"message": {"content": text}}]}
    if usage is not None:
        d["usage"] = usage
    return d


@pytest.fixture
def posted(monkeypatch):
    box = {}

    def fake_post(url, json=None, timeout=None):
        box.update(url=url, json=json, timeout=timeout)
        return FakeResponse(box.get("reply", ok_payload()))

    monkeypatch.setattr(requests, "post", fake_post)
    return box


def test_llamacpp_把system放在messages第一則(posted):
    """OpenAI 格式沒有獨立的 system 欄位，它是 messages 的第一則。"""
    LlamaCppLLM().complete(NEUTRAL, system="系統指令")

    assert posted["json"]["messages"][0] == {"role": "system", "content": "系統指令"}
    assert posted["json"]["messages"][1:] == NEUTRAL     # 中性格式原樣送出


def test_llamacpp_讀得到usage(posted):
    """實測 llama.cpp 會回 prompt_tokens / completion_tokens。
    「本機不計費」不等於「量不到」——地端更需要知道 prompt 有多長，
    才知道離 -c 的上限還有多遠。"""
    posted["reply"] = ok_payload(usage={"prompt_tokens": 36, "completion_tokens": 3})

    assert LlamaCppLLM().complete(NEUTRAL, system="s").usage == Usage(36, 3)


def test_llamacpp_沒有usage就當零(posted):
    """別的 OpenAI 相容 server 不一定回 usage，少了不該讓整次對話失敗。"""
    posted["reply"] = ok_payload()
    assert LlamaCppLLM().complete(NEUTRAL, system="s").usage == Usage(0, 0)


def test_llamacpp_回傳格式不對就丟例外(posted):
    posted["reply"] = {"error": "什麼鬼"}
    with pytest.raises(RuntimeError, match="非預期"):
        LlamaCppLLM().complete(NEUTRAL, system="s")


def test_llamacpp_溫度會送出去(posted):
    LlamaCppLLM().complete(NEUTRAL, system="s", temperature=0.9)
    assert posted["json"]["temperature"] == 0.9


def test_llamacpp_翻譯連線失敗():
    llm = LlamaCppLLM(url="http://localhost:9999/x")
    assert "9999" in llm.explain(requests.exceptions.ConnectionError())
    assert "載入" in llm.explain(requests.exceptions.Timeout())
    assert llm.explain(ValueError("誰知道")) is None
