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

# ── 待遇 ─────────────────────────────────────────────────────────────
# 薪資是唯一一個要「比大小」的欄位，所以不能像其他 facet 那樣用詞彙表比對。
# 實際的寫法一份一個樣（30 個職缺就有這些）：
#     面議 / 待遇面議 / 依學經歷及專業能力面議
#     月薪30,000元 / 月薪$36,000~41,000 (面議) / 月薪 NT$36,000–42,000，依…核定
#     月薪 36,000 ~ 43,000 元（正職） / 實習時薪 195 ~ 220 元
#     年薪600,000以上
#     時薪 NT$ 400 ~ 800 元或 專案論件計酬（依經驗面議）
# 所以要吃：NT$／$／逗號／元、~ ～ - – — 至 到 六種範圍符號、「以上」沒有天花板，
# 以及「月薪在前、實習時薪在後」這種一欄兩個數字。
PAY = re.compile(r"(月薪|年薪|時薪)\s*(?:NT)?\s*\$?\s*([\d,]+)"
                 r"(?:\s*[~～\-–—至到]\s*(?:NT)?\s*\$?\s*([\d,]+))?"
                 r"\s*元?\s*(以上|起)?")


def _int(s):
    return int(s.replace(",", ""))


def pay_of(text):
    """把「待遇」欄換算成可以比大小的月薪區間 [下限, 上限]。

    上限 None＝「以上」，沒有天花板。整個回 [] 代表**判斷不了**——面議、
    只給時薪、論件計酬都是這一類。

    ⚠ 「判斷不了」不等於「不符合」，而這是這個欄位最重要的一件事：過濾時
      判斷不了的會被整個排除，所以呼叫端一定要把被排除的數量講出來（見
      pay_caveat），否則清單看起來完整、其實漏掉一半——那比答不出來更糟。

    時薪不換算成月薪：不知道一個月排幾小時，乘一個猜的數字等於偽造資料。
    一欄同時有月薪和實習時薪時，取先出現的月薪（實測都是月薪寫在前面）。
    """
    for unit, lo, hi, unbounded in PAY.findall(text or ""):
        if unit == "時薪":
            continue
        low = _int(lo)
        high = None if unbounded else (_int(hi) if hi else low)
        if unit == "年薪":
            low //= 12
            high = high // 12 if high is not None else None
        return [low, high]
    return []


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
        # 空的 list 代表「判斷不了」，跟上面那些「資料沒寫」是同一種誠實。
        "pay": pay_of(meta.get("待遇", "")),
    }


def known_districts(docs):
    """把語料裡出現過的區名蒐集起來，當查詢時的詞彙表。

    區名在文件裡有「區」字可以當錨點，但問句裡沒有（「內湖的職缺」），
    所以查詢側沒辦法用同一個正規表示式，得靠詞彙表比對。
    """
    return sorted({d for doc in docs for d in doc.facets.get("district", [])})


# 問句裡的門檻方向。⚠ 「不超過」裡面含有「超過」、「不低於」裡面含有「低於」,
# 所以比對要照**出現位置**取最早的那個（否定詞在前，位置比較早就會贏），
# 不能照表的順序找到就算——那會把「不超過五萬」判成「超過五萬」。
PAY_ABOVE = ("以上", "超過", "至少", "大於", "高於", "多於", "起跳", "不低於", "破")
PAY_BELOW = ("以下", "以內", "低於", "少於", "小於", "不到", "不超過", "不滿")
PAY_KEYS = ("pay_min", "pay_max")

CN = {"一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5,
      "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
# 「五萬」「5萬」「三萬五」「5.5萬」都要認；「萬」後面那一個字是千位。
WAN = re.compile(r"(\d[\d,]*(?:\.\d+)?|[一二兩三四五六七八九十]+)\s*萬\s*"
                 r"([\d一二兩三四五六七八九])?")
# 沒有「萬」就要求至少四位數：避免把「前 5 名」「k=5」這種數字當成薪水。
PLAIN = re.compile(r"(\d[\d,]{3,})\s*元?")


def _cn(s):
    """中文或阿拉伯數字 → 數值。認不得就 None（於是不會組出過濾條件）。"""
    if not s:
        return None
    if s[0].isdigit():
        return float(s.replace(",", ""))
    if "十" in s:
        tens, _, ones = s.partition("十")
        return (CN.get(tens, 1) if tens else 1) * 10 + (CN.get(ones, 0) if ones else 0)
    return CN.get(s)


def _amount(q):
    m = WAN.search(q)
    if m:
        n = _cn(m.group(1))
        if n is None:
            return None
        return int(n * 10000 + (_cn(m.group(2)) or 0) * 1000)   # 「三萬五」的五是千位
    m = PLAIN.search(q)
    return _int(m.group(1)) if m else None


def _direction(q):
    """問的是「以上」還是「以下」。取出現位置最早、同位置取最長的那個。"""
    best = None
    for key, words in (("pay_min", PAY_ABOVE), ("pay_max", PAY_BELOW)):
        for w in words:
            i = q.find(w)
            if i >= 0 and (best is None or (i, -len(w)) < best[:2]):
                best = (i, -len(w), key)
    return best[2] if best else None


def _pay_filter(q):
    """從問句抓出薪資門檻。抓不到就回空的——退回純向量檢索，跟以前一樣。

    ⚠ 問「時薪超過 300」時**刻意不組條件**：職缺那邊的時薪換不成月薪
      （見 pay_of），硬比就是拿兩種單位相減。寧可退回舊行為，也不要給出
      一個有依據的樣子卻是錯的答案。
    """
    if not any(w in q for w in ("薪", "待遇", "月收", "年收")):
        return {}
    if "時薪" in q:
        return {}
    key, amount = _direction(q), _amount(q)
    if key is None or amount is None:
        return {}       # 「薪水多少」「薪水五萬」沒有方向，不猜
    if "年薪" in q or "年收" in q:
        amount //= 12   # 門檻也要換成月薪才比得起來
    return {key: amount}


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
    f.update(_pay_filter(q))        # 薪資是唯一要比大小的條件，見 _pay_filter
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
    parts = []
    for k, v in filters.items():
        if k in LABEL:
            parts.append("%s＝%s"
                         % (LABEL[k], "／".join(v) if isinstance(v, list) else "是"))
        elif k in PAY_KEYS:
            parts.append("月薪 %s 元%s"
                         % (format(v, ","), "以上" if k == "pay_min" else "以下"))
    return "、".join(parts)


def pay_caveat(unknown):
    """待遇判斷不了的那些職缺，數量一定要講出來。

    它們被過濾整個排除了（見 _pay_ok），而模型只看得到留下來的那幾筆。不講
    的話它會用一份少了一半的清單講出「只有這些」——上線就是這樣錯的：問
    「哪些公司薪水超過五萬」，30 個職缺裡 18 個寫面議或只給時薪，模型拿著
    剩下的講得斬釘截鐵。看起來完整卻不完整，比明說「有幾筆判斷不了」危險。
    """
    return ("另有 %d 個職缺的待遇寫「面議」或只給時薪，判斷不了，不在這份清單裡"
            "——回答時要把這件事一併告訴使用者。" % unknown) if unknown else ""


def _pay_ok(pay, key, want):
    """薪資門檻。判斷不了的（面議、只給時薪）一律不符合——寧可漏，不要編。

    比的是區間跟門檻有沒有重疊，不是比某一個端點：
      「月薪 45,000~60,000」問「五萬以上」→ 符合（上限碰得到）
      「年薪 600,000 以上」→ 換算成月薪 50,000 起、沒有天花板 → 符合
    """
    if not pay:
        return False
    low, high = pay
    if key == "pay_min":
        return high is None or high >= want
    return low <= want


def match(facets, filters):
    """一筆資料符不符合過濾條件。多值欄位只要有交集就算符合。"""
    for key, want in filters.items():
        if key in PAY_KEYS:
            if not _pay_ok(facets.get("pay"), key, want):
                return False
            continue
        got = facets.get(key)
        if key == "remote":
            if not got:
                return False
        elif not set(want) & set(got or []):
            return False
    return True
