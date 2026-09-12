# api/main.py（FastAPI 後端）
#   啟動：uvicorn api.main:app --reload
#   文件：http://localhost:8000/docs
#
#   這一層很薄，刻意的：它只做「HTTP 的事」——驗參數、把 Doc 轉成 Source、
#   把 provider 的例外翻成狀態碼。真正的邏輯還是在 shared/chat_bot.py，
#   所以 CLI、Streamlit、這個 API 三條路跑的是同一份程式。
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

import providers
from api import deps
from api.deps import BotFactoryDep
from api.schemas import ChatRequest, ChatResponse, Health, Message, Source, Usage
from shared.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 預熱：把 API_WARM 列到的那幾邊先建好索引，第一個使用者才不用等。
    # 預設是空的（全部延遲）——地端預熱要載 e5 再算 140 塊，只用雲端的人
    # 不該為此多等。
    for side in settings().warm_sides():
        deps.bot_for(side)
    yield


app = FastAPI(
    title="RAG Chat-bot API",
    version="0.1.0",
    summary="媒合會職缺查詢。雲端（Gemini）與地端（llama.cpp）同一組端點。",
    lifespan=lifespan,
)


@app.get("/health", response_model=Health)
def health():
    return Health(status="ok", sides=list(providers.SIDES), loaded=deps.loaded())


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, factory: BotFactoryDep):
    """檢索 + 生成。

    history 進來什麼樣、回去就是更新後的樣子——伺服器不記任何東西，
    理由見 api/schemas.py 開頭。
    """
    try:
        bot = factory(req.side)
    except Exception as e:      # 例如雲端沒設金鑰
        raise HTTPException(503, f"{req.side} 這一邊起不來：{e}") from e

    history = [m.model_dump() for m in req.history]      # ask() 會就地更新這個 list
    try:
        reply, hits, usage = bot.ask(req.question, history, k=req.k)
    except Exception as e:
        raise _as_http(bot, e) from e

    return ChatResponse(
        reply=reply,
        sources=[Source(code=d.group, label=d.label, score=s) for d, s in hits],
        usage=Usage(prompt=usage.prompt, output=usage.output),
        history=[Message(**m) for m in history],
    )


def _as_http(bot, exc):
    """把 provider 的例外翻成狀態碼。

    llm.explain() 認得的都是「上游的可預期狀況」——server 沒開、過載、配額
    用完。那些是 429/503，不是 500；500 要留給「我們自己壞了」，不然呼叫端
    沒辦法判斷該不該重試。
    """
    message = bot.llm.explain(exc)
    if message is None:
        return HTTPException(500, f"生成失敗：{exc}")
    return HTTPException(429 if getattr(exc, "code", None) == 429 else 503, message)
