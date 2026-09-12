# ingest.py（職缺 PDF → 可檢索的塊）
#
#   這些解析函式原本在 tools/build_jobs.py，是純離線腳本。現在上傳 API 也要用
#   （執行期把使用者丟進來的 PDF 解析成塊），所以搬到 shared/——
#   而且順便擺脫 build_jobs.py 模組層的 sys.stdout.reconfigure 副作用。
#
#   ⚠ 只吃「媒合會職缺表」那個格式的 PDF，不是通用文件載入器。
#     這是刻意的：整個檢索設計都綁在職缺的結構上——group 是代號、facets 從
#     地點與工作性質推出來、label 以代號開頭、SYSTEM 要求模型附上代號。
#     丟一份通用 Markdown 進來會產生沒有代號也沒有 facets 的塊，那些契約會
#     整個垮掉。要支援通用文件是另一個設計題，不是多加一個 loader。
import hashlib
import re
import unicodedata

import pypdfium2 as pdfium

from shared.facets import derive

# ── 清理：以下每一條都是實際掃過這批 PDF、逐處確認後才加的 ──────────────
FIXES = {
    "\ufffe": "-",    # 行尾連字號被 pdfium 抽成非字元。全庫 4 處，逐一比對原圖確認：
                      # LEOSYS-AILAB-SEC02 / end-to-end / cross-functional / Retrieval-Augmented
    "\uf0b2": "\u2022",   # 符號字型的項目符號，落在私人造字區（大世科的福利條列）
    "\uff61": "\u3002",   # 半形句號
}
# 康熙部首區。「⼀」(U+2F00) 跟漢字「一」(U+4E00) 長得一模一樣但碼位不同——
# 核流那份的「⼀個月內」就是這個，使用者打正常的「一」會完全搜不到。
KANGXI = range(0x2F00, 0x2FE0)
# 全形數字與英文字母。「有１年以上」的１是 U+FF11，使用者打「1年」搜不到。
# 只轉數字與字母：連全形標點一起轉的話，中文的（）會變成()，看起來會怪。
FULLWIDTH = (list(range(0xFF10, 0xFF1A)) + list(range(0xFF21, 0xFF3B))
             + list(range(0xFF41, 0xFF5B)))


def clean(t):
    for bad, good in FIXES.items():
        t = t.replace(bad, good)
    return "".join(unicodedata.normalize("NFKC", c)
                   if ord(c) in KANGXI or ord(c) in FULLWIDTH else c
                   for c in t)


def page_text(pdf):
    """逐頁抽文字，順手砍掉頁首的頁碼。

    頁碼在每一頁文字的「開頭」，直接串接的話會黏到前一頁的尾巴
    （例如「…願意嘗試開發實際可用的小工具或流程3」那個 3）。
    只有當開頭的數字剛好等於頁次時才砍，避免誤傷真的以數字開頭的內容。
    """
    out = []
    for i in range(len(pdf)):
        t = pdf[i].get_textpage().get_text_range()
        m = re.match(r"\s*(\d{1,3})\s*\r?\n", t)
        if m and int(m.group(1)) == i + 1:
            t = t[m.end():]
        out.append(t)
    return "".join(out)


CJK = r"\u4e00-\u9fff\u3000-\u303f\uff00-\uffef"


def join_lines(v):
    """把跨行的值接回一行。

    中文字之間不能補空白——PDF 的窄欄位會在詞中間斷行，補了空白就變成
    「工 作經驗」，使用者搜「工作經驗」就 match 不到。英文之間則要留空白。
    """
    v = re.sub(r"[ \t]*\r?\n[ \t]*", "\n", v.strip())
    v = re.sub(r"(?<=[" + CJK + r"])\n(?=[" + CJK + r"])", "", v)
    return v.replace("\n", " ")


# ── 解析：欄位順序在 11 份 PDF 裡是固定的，照順序往前找比自由比對標籤穩健 ──
ORDER = ["公司", "代號", "職缺", "地點", "工作性質", "待遇", "職務類別", "可上班日",
         "學歷要求", "工作經歷", "語文條件", "擅長工具", "工作內容", "應徵要求"]
TAIL = ["相關福利", "福利待遇", "公司福利", "福利"]      # 最後一欄各家叫法不同
SHORT = ORDER[:12]                                      # 前 12 欄短，合成「基本資料」塊
JOB_HEAD = re.compile(r"公司[ \t]+\S[^\r\n]{0,40}?[ \t]+代號[ \t]+\S+")


def label_re(label):
    """標籤自己也可能被換行切斷（臻至科技的欄位窄，「可上班日」存成「可上\n班日」），
    所以每個字之間都允許少量空白。限 0~4 個避免跨太遠誤配。"""
    return re.compile(r"\s{0,4}".join(map(re.escape, label)) + r"[ \t]*")


def parse_job(seg):
    marks, pos = [], 0
    for lab in ORDER:
        m = label_re(lab).search(seg, pos)
        if m:
            marks.append((lab, m.start(), m.end()))
            pos = m.end()
    for lab in TAIL:
        m = label_re(lab).search(seg, pos)
        if m:
            marks.append(("福利", m.start(), m.end()))
            break
    out = {}
    for i, (lab, _, e) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else len(seg)
        out[lab] = join_lines(seg[e:end])
    return out


def split_long(text, limit=420):
    """塊太長就再切。先找條列或句號這種語意邊界，真的沒有才硬切。
    420 是抓 512 的安全邊界——中文約一字一 token，留空間給 e5 的 "passage: " 前綴。"""
    if len(text) <= limit:
        return [text]
    parts, buf = [], ""
    for piece in re.split(r"(?<=[\u3002\n])|(?=\d+[.\u3001])|(?=\u2022)", text):
        if len(buf) + len(piece) > limit and buf:
            parts.append(buf.strip())
            buf = ""
        buf += piece
    if buf.strip():
        parts.append(buf.strip())
    return [p for p in parts if p]



def docs_from_pdf(data, source=""):
    """一份職缺 PDF 的位元組 → (塊清單, 職缺數)。

    塊是 dict，形狀跟 shared/knowledge.py 的 Doc 一致。一個 PDF 可能有好幾個
    職缺，每個職缺再切成好幾塊——切塊是為了繞過 embedding 的 512 token 上限。
    """
    pdf = pdfium.PdfDocument(data)
    txt = clean(page_text(pdf))
    starts = [m.start() for m in JOB_HEAD.finditer(txt)]

    docs, jobs = [], 0
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(txt)
        f = parse_job(txt[s:e])
        code = f.get("代號", "")
        if not code or code.startswith("："):      # 公司簡介誤判成職缺
            continue
        jobs += 1

        meta = {k: f[k] for k in SHORT if f.get(k)}
        # 代號放最前面：完整精確的清單要由程式給，不能指望模型複述。
        # 實測 3B 模型會自己決定格式、把代號丟掉，20 筆也只列得出 16~17 筆。
        label = " · ".join(x for x in (code, f.get("公司"), f.get("職缺"),
                                       f.get("地點")) if x)
        full = "\n".join("%s：%s" % (k, v) for k, v in f.items() if v)

        # 基本資料一塊，長欄位各自一塊；太長的再往下切
        chunks = [("基本", "　".join("%s %s" % (k, f[k])
                                     for k in SHORT if f.get(k)))]
        for k in ("工作內容", "應徵要求", "福利"):
            if not f.get(k):
                continue
            for j, part in enumerate(split_long(f[k])):
                name = "%s%s" % (k, j + 1 if j else "")
                chunks.append((name, "%s %s %s" % (f.get("職缺", ""), k, part)))

        fac = derive(f)          # 正規化成可精確比對的分類，見 shared/facets.py
        for name, text in chunks:
            docs.append({
                "id": "%s#%s" % (code, name), "text": text, "label": label,
                "full": full, "meta": meta, "facets": fac, "group": code,
                "source": source,
            })
    return docs, jobs


def version_of(data):
    """檔案內容的指紋。同一份檔案重傳就是同一個版本，不必重新切塊與算向量。"""
    return hashlib.sha256(data).hexdigest()[:16]
