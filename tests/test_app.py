"""shared/app.py 的純函式。

Streamlit 的畫面本身不測（要測就得起一個瀏覽器），但這兩個函式跟 UI 無關：
一個決定排版對不對，一個決定來源會不會標到別人的回答底下。兩個都出過事。
"""
from shared.app import _align, _hard_breaks


# ── 排版：markdown 會把單一換行吃掉 ──────────────────────────────────
def test_單一換行要補成markdown的硬換行():
    """模型列職缺時一行一個欄位。實測一則回答有 36 個換行，畫面上一個都
    看不到——二十筆職缺糊成一大坨。"""
    assert _hard_breaks("代號：A-01\n職缺名稱：X") == "代號：A-01  \n職缺名稱：X"


def test_串流逐段套用跟整段套用結果一樣():
    """純字元替換，所以不必擔心換行剛好被切在兩個 token 中間。
    用正規表示式做「只換單一換行」就會有這個問題。"""
    text = "台北有：\n\n代號：A-01\n職缺：X\n\n代號：B-02\n職缺：Y"

    assert "".join(_hard_breaks(c) for c in text) == _hard_breaks(text)


def test_沒有換行的回答不會被動到():
    assert _hard_breaks("資料裡沒有。") == "資料裡沒有。"


# ── 對齊：來源與警示都不能標到別人的回答底下 ─────────────────────────
def test_align_一次接上兩格user那格是None():
    assert _align([], ["來源"], 2) == [None, ["來源"]]
    assert _align([None, "a"], "b", 4) == [None, "a", None, "b"]


def test_align_history被截短時跟著切掉前面():
    """地端有 history 上限，後端會就地截短。不跟著切的話，Streamlit 那個
    zip(strict=True) 會直接炸——那是刻意的，寧可報錯也不要默默錯位。"""
    prev = [None, "a", None, "b"]

    assert _align(prev, "c", 4) == [None, "b", None, "c"]


def test_align_history清空就一起清空():
    assert _align([None, "a"], "b", 0) == []
