# api/schemas.py（請求與回應的形狀）
#
#   ⚠ 這個 API 是「無狀態」的：history 由呼叫端帶進來、原樣帶回去，
#     伺服器什麼都不記。
#
#   這不是偷懶，是刻意避開一個很容易踩的坑：把對話歷史做成模組層單例的話，
#   整個 process 共用一份 —— 兩個使用者會看到彼此的對話，任一人清空就清掉
#   所有人的。改成「綁在連線上」可以解決，但那要先有連線的概念；
#   REST 這一層根本不需要，把 history 交給呼叫端保管最單純：
#   併發永遠不會互相污染，伺服器可以隨時重啟，也能水平擴充。
#
#   Streamlit 那側本來就把 history 放在 st.session_state，剛好對得上。
from typing import Literal

from pydantic import BaseModel, Field

Side = Literal["cloud", "onperm"]


class Message(BaseModel):
    """中性格式，跟 shared/llm.py 用的是同一種。"""

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list, max_length=100)
    side: Side = "cloud"
    k: int | None = Field(default=None, ge=1, le=50)


class Source(BaseModel):
    """一筆檢索結果。

    這是「真正可靠的完整清單」——模型的散文只當摘要看（實測 3B 模型被要求
    列舉 20 筆時只列得出 16~17 筆，而且常常不照指示附代號）。所以 code
    一定要獨立成欄位，不能指望前端去散文裡撈。
    """

    code: str            # 職缺代號，應徵表單要填這個
    label: str
    score: float


class Usage(BaseModel):
    prompt: int
    output: int          # 雲端這個數字含思考 token——看不到但會計費


class ChatResponse(BaseModel):
    reply: str
    sources: list[Source]
    usage: Usage
    history: list[Message]      # 更新後的，呼叫端直接拿去覆蓋自己那份


class Health(BaseModel):
    status: Literal["ok"]
    sides: list[Side]           # 支援哪幾邊
    loaded: list[Side]          # 已經建好索引的（第一次呼叫才建，所以會變）


class ErrorBody(BaseModel):
    detail: str
