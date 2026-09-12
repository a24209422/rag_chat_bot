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
    loaded: list[Side]          # 索引已經算好的（第一次用到才算，所以會變）
    chunks: dict[Side, int]     # 各邊索引裡有幾塊
    documents: int              # 上傳的文件數（不含基礎語料）


class ErrorBody(BaseModel):
    detail: str


class DocumentOut(BaseModel):
    """一份上傳的文件。

    「基礎語料」（建置階段由 tools.build_jobs 產生的那批）不在這裡，
    也不能經由 API 刪除——它是這個服務的底線，不是使用者管理的東西。
    """

    doc_id: str
    filename: str
    content_type: str
    size: int
    version: str          # 檔案內容的指紋。同名不同版 = 更新
    jobs: int             # 這份 PDF 解析出幾個職缺
    created_at: str


class IndexChange(BaseModel):
    added: int = 0
    removed: int = 0


class UploadResult(BaseModel):
    document: DocumentOut
    # 只有「已經建好索引」的那幾邊會當場更新；還沒載入的會在第一次用到時
    # 自己跟 registry 對齊，所以不必在這裡等它。
    index: dict[Side, IndexChange]


class DeleteResult(BaseModel):
    doc_id: str
    index: dict[Side, IndexChange]
