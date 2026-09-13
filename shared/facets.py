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
#
# 這張表列得比現有語料多（目前只出現台北、新北、新竹、台南四個）。多列是
# 安全的：比對方式是「這個詞有沒有出現在文字裡」，沒出現的城市不會憑空
# 冒出來。而且下面的 REGION 要能表達「南部」的完整意思，就不能只列
# 語料裡剛好有的那幾個。
CITY = {
    "基隆": ["基隆"],
    "台北": ["台北", "臺北"],
    "新北": ["新北"],
    "桃園": ["桃園"],
    "新竹": ["新竹"],
    "苗栗": ["苗栗"],
    "台中": ["台中", "臺中"],
    "彰化": ["彰化"],
    "南投": ["南投"],
    "雲林": ["雲林"],
    "嘉義": ["嘉義"],
    "台南": ["台南", "臺南"],
    "高雄": ["高雄"],
    "屏東": ["屏東"],
    "宜蘭": ["宜蘭"],
    "花蓮": ["花蓮"],
    "台東": ["台東", "臺東"],
}

# 區域詞。**只用在查詢側**，職缺的 facets 不會多一個 region 欄位——
# 「南部」不是資料裡的屬性，是問句裡的說法。展開成城市之後就沿用既有的
# city 過濾，derive() 與 match() 一行都不用改。
#
# 為什麼需要它：問「南部有哪些職缺」時 parse_query 抽不出任何條件，於是
# 退回純向量檢索，撈回來的是語意相近但地點完全不對的職缺（實測撈到
# JETGO-CONTENT-01、Kneron-01、T504.4…，一個南部的都沒有），模型便誠實
# 回答「資料裡沒有」。錯的是檢索，不是生成。
REGION = {
    "北部": ["基隆", "台北", "新北", "桃園", "新竹"],
    "中部": ["苗栗", "台中", "彰化", "南投", "雲林"],
    "南部": ["嘉義", "台南", "高雄", "屏東"],
    "東部": ["宜蘭", "花蓮", "台東"],
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
    # 區域詞展開成城市，跟直接寫城市名的結果合併。用 dict.fromkeys 去重又
    # 保留順序——「南部或台南的職缺」不該讓台南出現兩次。
    city += [c for region, cities in REGION.items() if region in q for c in cities]
    if city:
        f["city"] = list(dict.fromkeys(city))
    kind = _hits(q, KIND)
    if kind:
        f["kind"] = kind
    degree = _hits(q, DEGREE)
    if degree:
        f["degree"] = degree
    if any(w in q.lower() for w in REMOTE):
        f["remote"] = True
    return f


def terms_in(q, districts=()):
    """問句裡實際出現的那幾個詞，照它們在問句中的順序，**保留使用者原本打的字**。

    跟 parse_query 是一對，但用途相反：parse_query 回正規化之後的值（「南部」
    變成嘉義／台南／高雄／屏東），給檢索用；這裡回原字，給「把省略式問句補成
    完整問句」用。

    為什麼一定要原字：實測（地端 Qwen2.5-3B，同樣的 history 與【資料】，
    只換【問題】那一行）

        南部呢？            6 次全部答「資料裡沒有」
        南部有哪些職缺？     6 次全對

    補成「嘉義、台南、高雄、屏東有哪些職缺？」是另一句話，沒被驗證過；
    補成「南部有哪些職缺？」才是量過的那一句。
    """
    words = ([*REGION] + [w for v in CITY.values() for w in v]
             + [w for v in KIND.values() for w in v]
             + [w for v in DEGREE.values() for w in v]
             + list(REMOTE) + list(districts))
    low = q.lower()
    found = {}                      # 出現位置 → 原字；同一個位置只留最長的
    for w in words:
        i = low.find(w.lower())
        if i >= 0 and len(w) > len(found.get(i, "")):
            found[i] = q[i:i + len(w)]      # 切原字，大小寫照使用者寫的
    return [found[i] for i in sorted(found)]


def as_question(terms):
    """把那些詞組成一個獨立可讀的問句。抽不出詞就回 None。

    「那薪水呢？」這種沒有任何可比對的詞，這條路幫不上——呼叫端要有別的
    退路（見 shared/chat_bot.py 的重問）。
    """
    return "、".join(terms) + "有哪些職缺？" if terms else None


LABEL = {"city": "地點", "district": "行政區", "kind": "工作性質",
         "degree": "學歷", "remote": "可遠端"}


def describe(filters):
    """把過濾條件講成一句人看得懂的話，要塞進 prompt 給模型看。

    為什麼需要：問「南部有哪些職缺」時 REGION 已經展開成
    嘉義／台南／高雄／屏東，而且篩出了全部符合的職缺——但模型不知道這件事。
    實測地端 Qwen2.5-3B 拿著台南的職缺、被問「南部有哪些」，6 次有 6 次回
    「資料裡沒有」：它不知道台南屬於南部。把展開結果直接寫給它看之後 6 次全對。

    ⚠ 這不是「叫模型聽話」的咒語，是補一塊它沒有的知識。兩者的差別在於
      前者靠運氣，後者可以解釋為什麼會有效。
    """
    return "、".join(
        "%s＝%s" % (LABEL[k], "／".join(v) if isinstance(v, list) else "是")
        for k, v in filters.items() if k in LABEL)


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
