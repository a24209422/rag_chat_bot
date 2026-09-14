"""對真實 data/jobs.json 的回歸測試。

docs/design-notes.md 裡那張 metadata 過濾的表是實測數字。這裡把它釘成測試——
而且完全不需要 embedding 模型：過濾是 facet 決定的，跟分數無關，
所以「撈回幾個」用 parse_query + match 就算得出來，整批不到一秒。

重建 jobs.json 之後這些數字會變，那時要一起更新（design-notes 也是）。
"""
from pathlib import Path

import pytest

from shared.facets import known_districts, match, parse_query
from shared.knowledge import default_docs

pytestmark = pytest.mark.corpus

if not Path("data/jobs.json").exists():
    pytest.skip("需要 data/jobs.json，先跑 python -m tools.build_jobs", allow_module_level=True)


@pytest.fixture(scope="module")
def docs():
    return default_docs()


def groups_matching(docs, question):
    """問句能撈回幾個「不同職缺」。等同 retrieve() 在有過濾條件時的行為。"""
    f = parse_query(question, known_districts(docs))
    assert f, f"「{question}」抽不出過濾條件，這題會退回純向量檢索"
    return {d.group for d in docs if match(d.facets, f)}


def test_語料規模(docs):
    assert len(docs) == 140                      # 塊
    assert len({d.group for d in docs}) == 30    # 職缺


def test_區名詞彙表(docs):
    assert known_districts(docs) == ["三重", "中和", "中山", "中正", "內湖",
                                     "大安", "永康", "萬華"]


@pytest.mark.parametrize("question, expected", [
    ("有哪些台北的職缺？",   20),
    ("內湖的職缺有哪些？",    8),
    ("三重的實習",           2),
    ("新北市有什麼工作？",    4),
    ("有沒有可以遠端的？",    4),
    ("有哪些實習機會？",      7),
    ("台北的實習有哪些？",    3),
])
def test_篩選型問句撈回完整清單(docs, question, expected):
    """這些數字就是 design-notes「metadata 過濾」那張表。"""
    assert len(groups_matching(docs, question)) == expected


def test_台與臺在真實資料裡都有(docs):
    """正規化如果壞掉，這兩個數字會對不起來——寫「臺北」的那 4 個會憑空消失。"""
    raw = " ".join(d.meta.get("地點", "") for d in docs)
    assert "臺北" in raw and "台北" in raw
    assert len(groups_matching(docs, "台北")) == 20      # 兩種寫法都算進來了


def test_每一塊都有group而且group數少於塊數(docs):
    """切塊是為了繞過 embedding 的 512 token 上限，所以塊一定比職缺多。"""
    assert all(d.group for d in docs)
    assert len({d.group for d in docs}) < len(docs)


def test_沒填地點的職缺誠實留空(docs):
    """有公司留著「【待填…】」範本。它們的 city 必須是空的，
    不能被範本裡的「例：台北市內湖區」汙染。"""
    placeholder = [d for d in docs if "待填" in d.meta.get("地點", "")]
    if not placeholder:
        pytest.skip("這批資料沒有待填範本")
    assert all(d.facets["city"] == [] for d in placeholder)
