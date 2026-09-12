"""shared/knowledge.py 的測試：Doc 的預設值，以及「import 不做事」這件事。"""
from pathlib import Path

import pytest

from shared.knowledge import Doc, default_docs, load_docs


def test_label與full沒給就退回text():
    """只給 text 的 Doc 要能直接用，不能有欄位是 None。"""
    d = Doc(id="A-01", text="內容")
    assert d.label == "內容"
    assert d.full == "內容"


def test_group沒給就用id():
    """沒有切塊的 Doc，它自己就是一個 group——去重邏輯才不會把它漏掉。"""
    assert Doc(id="A-01", text="x").group == "A-01"


def test_有給就不覆寫():
    d = Doc(id="A-01#工作內容", text="塊", label="標題", full="完整", group="A-01")
    assert (d.label, d.full, d.group) == ("標題", "完整", "A-01")


def test_text與full是兩個獨立欄位():
    """這是整個設計的關鍵：text 受 embedding 模型的 512 token 限制（超過會被
    靜默截斷），full 受生成模型的 context 限制（寬鬆得多）。一個職缺切成好幾塊
    各自 embed，但撈到任一塊都餵完整職缺。"""
    d = Doc(id="A-01#1", text="只有這一塊", full="整份職缺", group="A-01")
    assert d.text != d.full


def test_檔案不存在時的錯誤訊息要告訴使用者怎麼辦():
    """只說 FileNotFoundError 沒用，要講出下一步該跑什麼指令。"""
    with pytest.raises(FileNotFoundError, match="tools.build_jobs"):
        load_docs(Path("這個檔案不存在.json"))


def test_default_docs是延遲的():
    """import shared.knowledge 不會讀檔——沒有 data/jobs.json 也 import 得起來。
    這是 Stage 0 的重點之一：測試與 CI 需要能 import 而不需要資料。"""
    import shared.knowledge as K
    assert callable(K.default_docs)          # 是函式，不是 import 時就算好的常數
    assert not isinstance(getattr(K, "DOCS", None), list)   # 舊的模組層全域不該復活


def test_default_docs會快取():
    """整個 process 共用一份，雲端與地端才會拿到同一批 Doc 物件。"""
    if not Path("data/jobs.json").exists():
        pytest.skip("需要 data/jobs.json")
    assert default_docs() is default_docs()
