"""shared/facets.py 的測試。

這個模組是規則式的（不是叫模型抽），所以它的價值全在邊界處理正不正確。
以下每一個測試都對應一個「實際掃過那批職缺 PDF 才發現」的坑——
不是為了覆蓋率湊出來的案例。坑的來歷見 docs/design-notes.md 的「正規化踩到的三個坑」。
"""
import pytest

from shared.facets import (
    as_question,
    derive,
    describe,
    known_districts,
    match,
    parse_query,
    pay_caveat,
    terms_in,
)
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


# ── 區域詞：「南部」不是資料裡的屬性，是問句裡的說法 ──────────────────
@pytest.mark.parametrize("q, want", [
    ("南部有哪些職缺", {"嘉義", "台南", "高雄", "屏東"}),
    ("北部有什麼",     {"基隆", "台北", "新北", "桃園", "新竹"}),
    ("中部的職缺",     {"苗栗", "台中", "彰化", "南投", "雲林"}),
    ("東部的工作",     {"宜蘭", "花蓮", "台東"}),
])
def test_區域詞展開成城市(q, want):
    """不展開的話 parse_query 回空的，於是退回純向量檢索——實測問「南部有哪些
    職缺」撈回 JETGO-CONTENT-01、Kneron-01、T504.4，一個南部的都沒有，
    模型只好誠實回「資料裡沒有」。錯的是檢索不是生成。"""
    assert set(parse_query(q)["city"]) == want


def test_區域詞與城市名並存時不重複():
    assert parse_query("南部或台南的職缺")["city"].count("台南") == 1


def test_區域詞不會誤傷單純的城市問句():
    """「台北」裡沒有「北部」、「台中」裡沒有「中部」，確認沒有子字串誤配。"""
    assert parse_query("台北有職缺嗎") == {"city": ["台北"]}
    assert parse_query("台中有職缺嗎") == {"city": ["台中"]}


def test_語料沒有的城市抽得出條件但篩完是空的():
    """抽得出來、篩完沒有 → 這時候回「資料裡沒有」才是正確答案，
    跟展開之前「撈到一堆台北的職缺然後說沒有」是完全不同的兩件事。"""
    assert parse_query("嘉義有職缺嗎") == {"city": ["嘉義"]}


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


def test_describe_把條件講成人話():
    """這句話會塞進 prompt——模型不知道台南算南部，得把展開結果寫給它看。"""
    assert describe({"city": ["台南", "高雄"]}) == "地點＝台南／高雄"
    assert describe({"remote": True}) == "可遠端＝是"
    assert describe({"city": ["台北"], "kind": ["實習"]}) == "地點＝台北、工作性質＝實習"


# ── 補句用的原字（跟 parse_query 相反：它回正規化值，這裡回原字）────────
def test_terms_in_保留使用者原本打的字():
    """parse_query 把「南部」正規化成嘉義／台南／高雄／屏東，原字就丟了。
    但補句要用原字——「南部有哪些職缺？」才是實測 6/6 對的那一句，
    「嘉義、台南、高雄、屏東有哪些職缺？」是另一句話，沒被驗證過。"""
    assert terms_in("南部呢？") == ["南部"]
    assert terms_in("台北的實習呢") == ["台北", "實習"]
    assert terms_in("內湖呢", districts=["內湖", "三重"]) == ["內湖"]


def test_terms_in_照問句裡出現的順序():
    assert terms_in("實習，台北的") == ["實習", "台北"]


def test_terms_in_抽不到詞就是空的():
    """「那薪水呢？」沒有任何可比對的詞，補句這條路幫不上——
    呼叫端要有退路（見 chat_bot._retry_send）。"""
    assert terms_in("那薪水呢？") == []


def test_as_question_組成獨立可讀的問句():
    assert as_question(["南部"]) == "南部有哪些職缺？"
    assert as_question(["台北", "實習"]) == "台北、實習有哪些職缺？"
    assert as_question([]) is None


# ── 待遇：唯一一個要「比大小」的欄位 ──────────────────────────────────
@pytest.mark.parametrize("written, want", [
    ("月薪30,000元", [30000, 30000]),
    ("月薪$36,000~41,000 (面議)", [36000, 41000]),
    ("月薪 45,000 ~ 60,000 （面議）", [45000, 60000]),
    ("月薪 NT$36,000–42,000，依學經歷與能力核定", [36000, 42000]),   # 破折號不是減號
    ("年薪600,000以上", [50000, None]),            # 換算成月薪，而且沒有天花板
    ("月薪 36,000 ~ 43,000 元（正職） / 實習時薪 195 ~ 220 元", [36000, 43000]),
])
def test_待遇的每一種寫法都要認得(written, want):
    """這六種是那批 PDF 裡真的出現過的寫法，一份一個樣。"""
    assert derive({"待遇": written})["pay"] == want


@pytest.mark.parametrize("written", [
    "面議",
    "待遇面議",
    "依學經歷及專業能力面議",
    "時薪 NT$ 190 ~ 220 元（面議，享勞健保）",
    "時薪 NT$ 500 ~ 1,000 元 或 專案論件計酬（依經驗面議）",
    "",
])
def test_判斷不了的待遇要誠實留空(written):
    """30 個職缺裡有 18 個是這一類。時薪不換算成月薪——不知道一個月排幾小時，
    乘一個猜的數字等於偽造資料。空的代表「不知道」，不是「不符合」。"""
    assert derive({"待遇": written})["pay"] == []


@pytest.mark.parametrize("q, want", [
    ("哪些公司的薪水超過五萬", {"pay_min": 50000}),
    ("月薪四萬以上的職缺", {"pay_min": 40000}),
    ("薪水50000以上", {"pay_min": 50000}),
    ("待遇三萬五以上", {"pay_min": 35000}),        # 「萬」後面那個字是千位
    ("薪水不超過三萬五的", {"pay_max": 35000}),
    ("年薪超過六十萬", {"pay_min": 50000}),        # 門檻本身也要換成月薪
])
def test_問句裡的薪資門檻(q, want):
    assert parse_query(q) == want


def test_不超過不能被當成超過():
    """「不超過」裡面含有「超過」、「不低於」裡面含有「低於」。照詞表的順序
    找到就算的話方向會反過來——問「不超過三萬」會得到「三萬以上」的答案，
    剛好相反，而且看起來很合理。"""
    assert parse_query("薪水不超過三萬") == {"pay_max": 30000}
    assert parse_query("薪水不低於三萬") == {"pay_min": 30000}


def test_時薪不組條件():
    """職缺那邊的時薪換不成月薪（見 pay_of），硬比就是拿兩種單位相減。
    退回純向量檢索，跟沒有這個功能之前一樣——寧可沒答案，不要有依據的錯答案。"""
    assert parse_query("時薪超過300的職缺") == {}


def test_沒有方向或沒有數字就不猜():
    assert parse_query("薪水多少") == {}
    assert parse_query("薪水五萬") == {}     # 「五萬以上」還是「就是五萬」？不猜


def test_薪資條件跟其他條件併用():
    assert parse_query("台北月薪五萬以上的全職") == {
        "city": ["台北"], "kind": ["全職"], "pay_min": 50000}


def test_match_薪資比的是區間有沒有碰到門檻():
    assert match({"pay": [45000, 60000]}, {"pay_min": 50000})    # 上限碰得到
    assert match({"pay": [50000, None]}, {"pay_min": 50000})     # 「以上」沒有天花板
    assert not match({"pay": [36000, 41000]}, {"pay_min": 50000})
    assert match({"pay": [30000, 30000]}, {"pay_max": 35000})
    assert not match({"pay": [45000, 60000]}, {"pay_max": 35000})


def test_match_判斷不了的一律不符合():
    """寧可漏，不要編。代價是清單不完整，所以呼叫端一定要講出漏了幾筆。"""
    assert not match({"pay": []}, {"pay_min": 50000})
    assert not match({}, {"pay_min": 50000})      # 還沒重算過的舊資料也不能炸


def test_describe_薪資也要講成人話():
    assert describe({"pay_min": 50000}) == "月薪 50,000 元以上"
    assert describe({"pay_max": 35000}) == "月薪 35,000 元以下"


def test_pay_caveat_把被濾掉的數量講出來():
    """上線就是這樣錯的：問「哪些公司薪水超過五萬」，30 個職缺有 18 個寫面議
    或只給時薪，模型拿著剩下的 12 個講「只有這些」。看起來完整卻不完整，
    比明說「有幾筆判斷不了」危險得多。"""
    assert "18" in pay_caveat(18)
    assert pay_caveat(0) == ""
