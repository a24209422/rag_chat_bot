"""tools/build_jobs.py 裡那些清理函式的測試。

跟 facets.py 一樣，這些規則每一條都是實際掃過那批 PDF、逐處比對原圖之後
才加的。它們壞掉的症狀都是「使用者打正常的字卻搜不到」——不會報錯，
所以只能靠測試釘住。
"""
import pytest

from tools.build_jobs import clean, join_lines, label_re, page_text, parse_job, split_long


# ── clean()：PDF 抽出來的字元有好幾種長得一樣但碼位不同 ──────────────
def test_康熙部首要正規化成漢字():
    """「⼀」(U+2F00) 跟「一」(U+4E00) 長得一模一樣但碼位不同。
    核流那份的「⼀個月內」就是這個——使用者打正常的「一」會完全搜不到。"""
    assert clean("\u2f00個月內") == "一個月內"


@pytest.mark.parametrize("raw, want", [
    ("有\uff11年以上", "有1年以上"),      # 全形數字 U+FF11
    ("\uff21I 工程師", "AI 工程師"),      # 全形字母 U+FF21
])
def test_全形數字與字母要轉半形(raw, want):
    """「有１年以上」的１是 U+FF11，使用者打「1年」搜不到。"""
    assert clean(raw) == want


def test_全形標點不要跟著轉():
    """只轉數字與字母。連全形標點一起轉的話，中文的（）會變成 ()，看起來會怪。"""
    assert clean("（台北）") == "（台北）"


@pytest.mark.parametrize("raw, want", [
    ("end\ufffeto\ufffeend", "end-to-end"),   # 行尾連字號被抽成非字元，全庫 4 處
    ("\uf0b2 福利", "\u2022 福利"),            # 符號字型的項目符號，落在私人造字區
    ("結束\uff61", "結束。"),                   # 半形句號
])
def test_已知的壞字元對照(raw, want):
    assert clean(raw) == want


# ── join_lines()：PDF 的窄欄位會在詞中間斷行 ──────────────────────────
def test_中文之間不能補空白():
    """補了空白就變成「工 作經驗」，使用者搜「工作經驗」就 match 不到。"""
    assert join_lines("工\n作經驗") == "工作經驗"


def test_英文之間要留空白():
    """英文詞之間本來就有空白，接掉會變成 machinelearning。"""
    assert join_lines("machine\nlearning") == "machine learning"


def test_中英交界保留空白():
    assert join_lines("會\nPython") == "會 Python"


def test_跨行的縮排會被吃掉():
    """縮排清掉，而且因為兩邊都是中文，換行也直接接掉——沒有多出來的空白。"""
    assert join_lines("  第一行  \n    第二行  ") == "第一行第二行"


# ── page_text()：砍頁碼 ────────────────────────────────────────────────
class FakePage:
    def __init__(self, text):
        self._text = text

    def get_textpage(self):
        return self

    def get_text_range(self):
        return self._text


class FakePdf:
    def __init__(self, *pages):
        self._pages = [FakePage(p) for p in pages]

    def __len__(self):
        return len(self._pages)

    def __getitem__(self, i):
        return self._pages[i]


def test_頁碼會被砍掉():
    """頁碼在每頁文字的開頭，不砍的話會黏到前一頁的尾巴
    （例如「…可用的小工具或流程3」那個 3）。"""
    assert page_text(FakePdf("1\n第一頁內容", "2\n第二頁內容")) == "第一頁內容第二頁內容"


def test_開頭數字不等於頁次就不要砍():
    """避免誤傷真的以數字開頭的內容。"""
    assert page_text(FakePdf("3\n年工作經驗")) == "3\n年工作經驗"


# ── split_long()：切到 embedding 吃得下 ───────────────────────────────
def test_夠短就不切():
    assert split_long("短", limit=10) == ["短"]


def test_優先在句號切():
    """先找條列或句號這種語意邊界，真的沒有才硬切。"""
    assert split_long("一二三。四五六。七八九。", limit=6) == ["一二三。", "四五六。", "七八九。"]


def test_切出來的每一塊都不是空的():
    parts = split_long("甲。乙。丙。丁。", limit=4)
    assert all(p.strip() for p in parts)
    assert "".join(parts) == "甲。乙。丙。丁。"


# ── label_re()：標籤自己也可能被換行切斷 ──────────────────────────────
def test_標籤被換行切斷也要找得到():
    """臻至科技的欄位窄，「可上班日」在 PDF 裡被存成「可上\n班日」。"""
    assert label_re("可上班日").search("可上\n班日 ")


def test_標籤不會跨太遠誤配():
    """每個字之間只允許 0~4 個空白，不然整份文件裡任兩個字都能湊成一個標籤。"""
    assert not label_re("代號").search("代" + " " * 20 + "號")


# ── parse_job()：欄位照順序往前找 ─────────────────────────────────────
FULL = ("公司 甲公司 代號 A-01 職缺 工程師 地點 台北市內湖區 工作性質 全職 "
        "待遇 面議 職務類別 軟體 可上班日 隨時 學歷要求 大學 工作經歷 不拘 "
        "語文條件 中文 擅長工具 Python 工作內容 寫程式 應徵要求 附作品集 ")


def test_依序抽出欄位():
    """欄位順序在 11 份 PDF 裡是固定的，照順序往前找比自由比對標籤穩健。"""
    got = parse_job(FULL)
    assert got["公司"] == "甲公司"
    assert got["代號"] == "A-01"
    assert got["地點"] == "台北市內湖區"
    assert got["工作性質"] == "全職"
    assert got["應徵要求"] == "附作品集"


@pytest.mark.parametrize("tail", ["相關福利", "福利待遇", "公司福利", "福利"])
def test_最後一欄各家叫法不同都要認(tail):
    """相關福利 / 福利待遇 / 公司福利 / 福利，統一收成「福利」。"""
    assert parse_job(FULL + tail + " 三節獎金")["福利"] == "三節獎金"


def test_欄位必須照ORDER的順序出現():
    """這是 parse_job 的前提，不是 bug，但一定要知道：標籤是「從上一個命中的
    位置往後找」，所以順序錯了就會漏。

    具體例子：ORDER 裡有「待遇」，尾欄叫法裡有「福利待遇」。欄位完整時
    「待遇」早在它自己的位置就被吃掉了，尾端的「福利待遇」才輪得到 TAIL 比對；
    但前面的欄位缺席時，「待遇」會在尾端命中，把「福利待遇」拆成兩半。

    換句話說：這個解析器綁定「這批 PDF 的欄位格式」。換一批格式不同的 PDF
    就要重寫 ORDER，而不是期待它自動適應。
    """
    缺欄位 = "公司 甲 代號 A-01 福利待遇 三節獎金"
    assert "福利" not in parse_job(缺欄位)          # 被 ORDER 的「待遇」吃掉了
    assert parse_job(缺欄位)["待遇"] == "三節獎金"
