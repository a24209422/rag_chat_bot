"""shared/app.py 的純函式。

Streamlit 的畫面本身不測（要測就得起一個瀏覽器），但這兩個函式跟 UI 無關：
一個決定排版對不對，一個決定來源會不會標到別人的回答底下。兩個都出過事。
"""
from shared.app import _align


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
