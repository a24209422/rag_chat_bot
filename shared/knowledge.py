# knowledge.py（雲端與地端共用的知識庫與系統指令）
#   這是兩邊「唯一」真正共用的東西。其餘看起來像的地方都不能合併：
#   history 格式一邊是 Gemini（parts/model）、一邊是 OpenAI（content/assistant），
#   ask() 的回傳個數也不同（雲端多一個 usage，地端不計費所以沒有）。
#
#   ⚠ 改了 DOCS 就要重跑 tools/probe_threshold.py。
#     cloud/rag.py 的 0.70 和 onperm/rag.py 的 0.84 都是對「這批」資料量出來的，
#     不是通用常數。資料換了、門檻沒跟著換，檢索就會失準。
from dataclasses import dataclass, field


@dataclass
class Doc:
    """一個檢索單位。

    關鍵在於 text 和 full 是兩個欄位，因為兩個長度上限是不同的東西：
      - text 拿去算向量，受 embedding 模型限制（e5-small 只吃 512 token，
        超過會被「靜默截斷」——不報錯，尾巴就是永遠檢索不到）
      - full 餵給生成模型，受 context 限制（Qwen2.5-3B 開 -c 4096，寬鬆得多）
    所以長文件的正確做法是：切小塊各自 embed，撈到任一塊就餵完整內容。
    現在的 FAQ 每筆都只有一句話，兩者相同，差異要等職缺那種長文進來才看得到。
    """

    id: str                                    # 唯一識別。手動給，不要自動編號——
    text: str                                  # 編號會在插入新資料時整批位移
    label: str = ""                            # 一行摘要：UI caption 與 probe_threshold 用
    full: str = ""                             # 餵生成模型的完整內容
    meta: dict = field(default_factory=dict)   # 結構化欄位（代號、公司、地點…），可空
    group: str = ""                            # 同源標記：同一份文件切出來的多塊共用一個

    def __post_init__(self):
        # 三個欄位沒填就退回 text / id，這樣單句資料寫起來跟改造前一樣短
        if not self.label:
            self.label = self.text
        if not self.full:
            self.full = self.text
        if not self.group:
            self.group = self.id


DOCS = [
    Doc("faq-refund",   "退款流程：商品收到後 7 天內可申請退款，需保持包裝完整。"),
    Doc("faq-shipping", "運費說明：訂單滿 1000 元免運，未滿加收 80 元。"),
    Doc("faq-hours",    "客服時間：週一至週五 09:00-18:00，例假日不營業。"),
    Doc("faq-member",   "會員等級：累積消費滿 5000 元升級為金卡，享 95 折。"),
    Doc("faq-warranty", "保固政策：電子產品保固一年，人為損壞不在範圍內。"),
]

SYSTEM = ("你是客服助理。只依據提供的【資料】回答，"     # 不講這句，模型會無視資料瞎掰
          "資料沒提到的就說「資料裡沒有」。")
