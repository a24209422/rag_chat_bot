# knowledge.py（雲端與地端共用的知識庫與系統指令）
#   這是兩邊「唯一」真正共用的資料。其餘看起來像的地方都不能合併：
#   history 格式一邊是 Gemini（parts/model）、一邊是 OpenAI（content/assistant），
#   ask() 的回傳個數也不同（雲端多一個 usage，地端不計費所以沒有）。
#
#   ⚠ 重建 data/jobs.json 之後要重跑 tools/probe_threshold.py。
#     兩邊 Retriever 的 min_score 都是對「這批」資料量出來的，不是通用常數。
#     資料換了、門檻沒跟著換，檢索就會失準。
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

JOBS = Path(__file__).resolve().parent.parent / "data" / "jobs.json"


@dataclass
class Doc:
    """一個檢索單位。

    關鍵在於 text 和 full 是兩個欄位，因為兩個長度上限是不同的東西：
      - text 拿去算向量，受 embedding 模型限制（e5-small 只吃 512 token，
        超過會被「靜默截斷」——不報錯，尾巴就是永遠檢索不到）
      - full 餵給生成模型，受 context 限制（Qwen2.5-3B 開 -c 8192，寬鬆得多）
    所以一個職缺會被切成好幾塊各自 embed，但撈到任一塊都餵完整職缺。
    """

    id: str                                    # 唯一識別，如 "ACO-01#工作內容"
    text: str                                  # 算向量用的那一塊
    label: str = ""                            # 一行摘要：UI caption 與 probe_threshold 用
    full: str = ""                             # 餵生成模型的完整職缺
    meta: dict = field(default_factory=dict)   # 結構化欄位原文（代號、公司、地點…）
    facets: dict = field(default_factory=dict)  # 正規化後的分類，用來精確過濾
                                               # （見 shared/facets.py）
    group: str = ""                            # 同源標記：同一個職缺切出來的多塊共用代號

    def __post_init__(self):
        if not self.label:
            self.label = self.text
        if not self.full:
            self.full = self.text
        if not self.group:
            self.group = self.id


def load_docs(path=JOBS):
    """讀 jobs.json。純函式，每次呼叫都重讀——測試要塞自己的知識庫就用這個。"""
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}。先跑：\n"
            f'  python -m tools.build_jobs "<職缺 PDF 資料夾>"')
    return [Doc(**d) for d in json.loads(path.read_text(encoding="utf-8"))]


@lru_cache(maxsize=1)
def default_docs():
    """預設知識庫，整個 process 共用一份。

    刻意是「函式 + 快取」而不是模組層的 DOCS = load_docs()：
      · import 這個模組不會讀檔，所以沒有 data/jobs.json 也 import 得起來
        （測試與 CI 需要這個；模組層讀檔的話連 import 都會炸）
      · 雲端與地端的 Retriever 預設都拿這一份，同一批 Doc 物件才比得出
        兩邊的差異——這是這個專案並排兩套實作的意義所在
    要換知識庫就別走這裡，直接把自己的 docs 傳進 Retriever(docs=...)。
    """
    return load_docs()


SYSTEM = ("你是媒合會職缺查詢助理。只依據提供的【資料】回答，"      # 不講這句，模型會無視資料瞎掰
          "資料沒提到的就說「資料裡沒有」。"
          "提到任何職缺時一定要附上它的代號——"                    # 回函表單要填代號，
          "應徵時要填在表單上。")                                  # 少了它使用者沒辦法報名
