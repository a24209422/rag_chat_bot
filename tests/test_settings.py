"""shared/settings.py。"""
from shared.settings import Settings, settings


def test_預設值():
    s = Settings(_env_file=None)               # 不要讀本機的 .env，測的是預設值
    assert s.gemini_chat_model == "gemini-flash-latest"
    assert s.gemini_embed_dim == 768
    assert s.retrieve_k == 5
    assert s.temperature == 0.2


def test_兩邊的history上限不同():
    """地端的 context 小要砍；雲端寬裕，砍了反而失去上下文。"""
    s = Settings(_env_file=None)
    assert s.onperm_history_limit == 11
    assert s.cloud_history_limit is None


def test_可以直接覆寫():
    """測試不必動環境變數。"""
    assert Settings(_env_file=None, temperature=0.9).temperature == 0.9


def test_環境變數會被讀到(monkeypatch):
    monkeypatch.setenv("TEMPERATURE", "0.42")
    assert Settings(_env_file=None).temperature == 0.42


def test_字串會被轉成正確型別(monkeypatch):
    """環境變數一律是字串，型別轉換是用 pydantic 的理由。"""
    monkeypatch.setenv("RETRIEVE_K", "9")
    s = Settings(_env_file=None)
    assert s.retrieve_k == 9 and isinstance(s.retrieve_k, int)


def test_settings有快取():
    assert settings() is settings()


def test_min_score刻意不在設定裡():
    """它不是旋鈕，是 probe_threshold 對「這個模型＋這批語料」量出來的結果。
    放進 .env 會讓它看起來可以隨手調——那正是最危險的誤解。
    它留在各自的 Retriever 類別上，跟量測過程的說明放在一起。"""
    assert not any("min_score" in f for f in Settings.model_fields)
