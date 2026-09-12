# facets.py（把自由文字的 metadata 正規化成可精確比對的分類）
#
#   為什麼需要這一步：純向量檢索答不好「篩選型」問題。問「有哪些台北的職缺」
#   時，語意相似度沒辦法保證「全部」——它只會給你最像的前 k 個。但地點、
#   工作性質這些欄位本來就是結構化的，直接篩就能保證完整。
#
#   問題是原始值很髒，不能直接當分類用：
#     地點  「台北市中正區」「臺北市內湖區」「台北或新竹」「全遠端 (Remote) / 彈性」
#     性質  「全職」「正職」「實習生／長期實習」「全職（應屆畢業生）或 兼職（實習生）」
#   所以在 build 階段（tools/build_jobs.py）先正規化，查詢時比對的是乾淨的值。
import re

# 台／臺 兩種寫法都有——實測 21 個用「台」、4 個用「臺」。
# 只比對一種會漏掉，所以先統一。
CITY = {
    "台北": ["台北", "臺北"],
    "新北": ["新北"],
    "桃園": ["桃園"],
    "新竹": ["新竹"],
    "台中": ["台中", "臺中"],
    "台南": ["台南", "臺南"],
    "高雄": ["高雄"],
}
KIND = {
    "全職": ["全職", "正職"],
    "實習": ["實習"],
    "兼職": ["兼職", "工讀", "part-time", "parttime"],
}
DEGREE = {
    "專科": ["專科", "五專", "二專"],
    "大學": ["大學", "學士", "大專", "在學"],
    "碩士": ["碩士", "研究所"],
}
REMOTE = ["遠端", "remote", "在家"]

# 有公司沒填地點，留著範本文字：「【待填：公司地點，例：台北市內湖區】」。
# 不砍掉的話會被裡面的「例：台北市內湖區」誤判成台北。
PLACEHOLDER = re.compile(r"【[^】]*待填[^】]*】")
# 括號裡是地標補充，判斷城市前要砍掉。
# 「新北市三重區（捷運台北橋站步行約 30 秒）」的「台北橋」是站名不是城市，
# 不砍的話這三個臻至科技的職缺會同時被判成台北和新北。
PAREN = re.compile(r"[（(][^）)]*[）)]")

# 區名用抽取而不是硬編一張全台區名表，新資料進來才能自動涵蓋。
# 必須緊跟在「市／縣」或城市名之後：不加這個錨點的話，「台北大安區」會被
# 抓成「北大安」。台灣的區名幾乎都是兩個字。
DISTRICT = re.compile(r"(?:市|縣|台北|臺北|新北|桃園|新竹|台中|臺中|台南|臺南|高雄)"
                      r"([一-鿿]{2})區")


def _hits(text, table):
    t = text.lower()
    return [key for key, words in table.items() if any(w in t for w in words)]


def derive(meta):
    """從原始 meta 算出乾淨的分類。build 階段呼叫一次，結果存進 jobs.json。

    全部都是多值的——一個職缺可能同時是遠端和台南（INT-01），
    也可能同時收全職和兼職（LEOSYS-AILAB-OT01）。
    抓不到就是空的，代表「資料沒寫」而不是「否」——中華創智那三個職缺的
    工作性質欄被填成了工作內容，那就誠實地留空，不要猜。
    """
    loc = PLACEHOLDER.sub("", meta.get("地點", ""))
    city_src = PAREN.sub("", loc)          # 城市看砍掉括號的版本
                                           # remote 看原文：「全遠端 (Remote)」括號內外都有
    # 性質同時看「工作性質」和職缺名稱：有些公司把性質寫在職缺名裡
    # （「AI創作系統實習生」「AI 編程兼職工程師」）
    kind_src = meta.get("工作性質", "") + " " + meta.get("職缺", "")
    return {
        "city": _hits(city_src, CITY),
        # 用 dict.fromkeys 去重又保留順序。抓不到就空的——
        # 「全遠端」「台北或新竹」「【待填…】」都沒有區級資訊，那就誠實留空。
        "district": list(dict.fromkeys(DISTRICT.findall(city_src))),
        "remote": any(w in loc.lower() for w in REMOTE),
        "kind": _hits(kind_src, KIND),
        "degree": _hits(meta.get("學歷要求", ""), DEGREE),
    }


def known_districts(docs):
    """把語料裡出現過的區名蒐集起來，當查詢時的詞彙表。

    區名在文件裡有「區」字可以當錨點，但問句裡沒有（「內湖的職缺」），
    所以查詢側沒辦法用同一個正規表示式，得靠詞彙表比對。
    """
    return sorted({d for doc in docs for d in doc.facets.get("district", [])})


def parse_query(q, districts=()):
    """從問句抓出過濾條件。

    規則式而不是叫模型抽：零延遲、可預測、出錯時看得出是哪條規則錯了。
    城市／性質／學歷的詞彙表就是上面那幾張，跟 derive() 共用所以不會對不起來；
    區名的詞彙表要由呼叫端給（見 known_districts），因為它是從資料長出來的。
    """
    f = {}
    dist = [d for d in districts if d in q]
    if dist:
        f["district"] = dist
    city = _hits(q, CITY)
    if city:
        f["city"] = city
    kind = _hits(q, KIND)
    if kind:
        f["kind"] = kind
    degree = _hits(q, DEGREE)
    if degree:
        f["degree"] = degree
    if any(w in q.lower() for w in REMOTE):
        f["remote"] = True
    return f


def match(facets, filters):
    """一筆資料符不符合過濾條件。多值欄位只要有交集就算符合。"""
    for key, want in filters.items():
        got = facets.get(key)
        if key == "remote":
            if not got:
                return False
        elif not set(want) & set(got or []):
            return False
    return True
