"""shared/facets.py 的測試。

這個模組是規則式的（不是叫模型抽），所以它的價值全在邊界處理正不正確。
以下每一個測試都對應一個「實際掃過那批職缺 PDF 才發現」的坑——
不是為了覆蓋率湊出來的案例。坑的來歷見 README 的「正規化踩到的三個坑」。
"""
import pytest

from shared.facets import derive, known_districts, match, parse_query
from shared.knowledge import Doc


# ── 坑 1：台／臺 兩種寫法都有 ──────────────────────────────────────────
@pytest.mark.parametrize("written", ["台北市內湖區", "臺北市內湖區"])
def test_台與臺都要認得(written):
    """實測 21 個職缺寫「台北」、4 個寫「臺北」。只比對一種會漏掉那 4 個。"""
    assert derive({"地點": written})["city"] == ["台北"]


# ── 坑 2：有公司沒填地點，留著範本文字 ────────────────────────────────
def test_待填範本不能被當成真的地點():
    """範本文字「【待填：公司地點，例：台北市內湖區】」裡面有「台北市內湖區」，
    不砍掉的話這個沒填地點的職缺會被誤判成台北。"""
    facets = derive({"地點": "【待填：公司地點，例：台北市內湖區】"})
    assert facets["city"] == []
    assert facets["district"] == []


# ── 坑 3：括號裡是地標不是城市 ────────────────────────────────────────
def test_括號裡的地標不能被當成城市():
    """「新北市三重區（捷運台北橋站步行約 30 秒）」的「台北橋」是站名。
    不砍括號的話，臻至科技那三個職缺會同時被判成台北和新北。"""
    facets = derive({"地點": "新北市三重區（捷運台北橋站步行約 30 秒）"})
    assert facets["city"] == ["新北"]
    assert facets["district"] == ["三重"]


def test_remote_看的是砍括號前的原文():
    """城市要看砍掉括號的版本，但 remote 不行——「全遠端 (Remote) / 彈性」
    的關鍵字括號內外都有，砍掉就抓不到了。"""
    assert derive({"地點": "全遠端 (Remote) / 彈性"})["remote"] is True


# ── 區名是抽出來的，不是硬編的 ────────────────────────────────────────
@pytest.mark.parametrize("loc", ["台北大安區", "台北市大安區"])
def test_區名必須有錨點(loc):
    """區名得緊跟在「市／縣」或城市名之後。少了這個錨點，
    「台北大安區」會被抓成「北大安」。"""
    assert derive({"地點": loc})["district"] == ["大安"]


def test_沒有區級資訊就誠實留空():
    """「全遠端」「台北或新竹」都沒有區級資訊，不要硬猜。"""
    assert derive({"地點": "台北或新竹"})["district"] == []


def test_known_districts_是從語料長出來的():
    """查詢側沒有「區」字可當錨點（問句是「內湖的職缺」），只能靠詞彙表比對。
    詞彙表由資料決定，所以重建 jobs.json 之後會自動更新。"""
    docs = [
        Doc(id="A", text="x", facets={"district": ["內湖", "大安"]}),
        Doc(id="B", text="y", facets={"district": ["內湖"]}),
        Doc(id="C", text="z", facets={}),          # 沒有 district 欄也不能炸
    ]
    assert known_districts(docs) == ["內湖", "大安"]


# ── 分類全部是多值的 ──────────────────────────────────────────────────
def test_一個職缺可以同時是遠端和某個城市():
    """INT-01 就是這樣。"""
    facets = derive({"地點": "台南（可全遠端 Remote）"})
    assert facets["city"] == ["台南"]
    assert facets["remote"] is True


def test_多個城市都要收():
    assert derive({"地點": "台北或新竹"})["city"] == ["台北", "新竹"]


def test_工作性質也看職缺名稱():
    """有些公司把性質寫在職缺名裡，工作性質欄反而是空的。"""
    assert derive({"工作性質": "", "職缺": "AI創作系統實習生"})["kind"] == ["實習"]


def test_抓不到就留空代表資料沒寫而不是否():
    """中華創智那三個職缺的「工作性質」欄被填成了工作內容。
    誠實留空，不要猜成全職。"""
    assert derive({"工作性質": "負責影像標註與資料整理"})["kind"] == []


def test_全職或兼職會同時收到():
    """LEOSYS-AILAB-OT01 的工作性質是「全職（應屆畢業生）或 兼職（實習生）」。

    注意這裡也會抓到「實習」——因為括號裡的「實習生」。這是已知的過度比對：
    多值本來就允許一個職缺屬於多類，而「實習」在這個情境下也不算全錯，
    所以維持現狀。這個測試是要把行為釘住，不是宣稱它完美。
    """
    kinds = derive({"工作性質": "全職（應屆畢業生）或 兼職（實習生）"})["kind"]
    assert set(kinds) == {"全職", "兼職", "實習"}


# ── 查詢側：從問句抽條件 ──────────────────────────────────────────────
@pytest.mark.parametrize("q, expected", [
    ("有哪些台北的職缺？",   {"city": ["台北"]}),
    ("內湖的職缺有哪些？",   {"district": ["內湖"]}),
    ("三重的實習",          {"district": ["三重"], "kind": ["實習"]}),
    ("有沒有可以遠端的？",   {"remote": True}),
    ("碩士以上的工作",       {"degree": ["碩士"]}),
    ("有哪些無人機相關的職缺？", {}),      # 抽不出條件 → 退回純向量檢索
])
def test_parse_query(q, expected):
    assert parse_query(q, districts=["三重", "內湖", "大安"]) == expected


def test_詞彙表外的區名不觸發過濾():
    """「板橋」沒出現在語料裡，所以不在詞彙表中，過濾不會觸發、退回純向量檢索。
    實測模型會正確回答「資料裡沒有」，所以這個邊界不需要額外處理。"""
    assert parse_query("板橋有工作嗎？", districts=["三重", "內湖"]) == {}


# ── 比對：多值只要有交集就算符合 ──────────────────────────────────────
def test_match_多值有交集就算符合():
    facets = {"city": ["台北", "新竹"], "kind": ["全職", "兼職"]}
    assert match(facets, {"city": ["新竹"]})
    assert match(facets, {"kind": ["兼職"]})
    assert not match(facets, {"city": ["高雄"]})


def test_match_多個條件要全部滿足():
    facets = {"city": ["台北"], "kind": ["實習"]}
    assert match(facets, {"city": ["台北"], "kind": ["實習"]})
    assert not match(facets, {"city": ["台北"], "kind": ["全職"]})


def test_match_remote_是布林不是清單():
    assert match({"remote": True}, {"remote": True})
    assert not match({"remote": False}, {"remote": True})


def test_match_欄位缺席就當不符合():
    """facets 裡沒有這個欄位（不是空清單，是根本沒有）也不能炸。"""
    assert not match({}, {"city": ["台北"]})
    assert match({}, {})                      # 沒有條件 = 全部符合
