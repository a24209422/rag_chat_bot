# api/main.py（FastAPI 後端）
#   啟動：uvicorn api.main:app --reload
#   文件：http://localhost:8000/docs
#
#   這一層很薄，刻意的：它只做「HTTP 的事」——驗參數、把 Doc 轉成 Source、
#   把 provider 的例外翻成狀態碼。真正的邏輯還是在 shared/chat_bot.py，
#   所以 CLI、Streamlit、這個 API 三條路跑的是同一份程式。
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import providers
from api import deps
from api.deps import BotFactoryDep, RegistryDep
from api.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResult,
    DocumentOut,
    Health,
    Message,
    Source,
    UploadResult,
    Usage,
)
from shared.ingest import docs_from_pdf, version_of
from shared.registry import doc_id_for
from shared.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 預熱：把 API_WARM 列到的那幾邊先建好索引，第一個使用者才不用等。
    # 預設是空的（全部延遲）——地端預熱要載 e5 再算 140 塊，只用雲端的人
    # 不該為此多等。
    #
    # ⚠ 一定要呼叫 doc_vecs()。只做 bot_for(side) 的話只是把物件建出來，
    #   而 Retriever 是延遲的——索引還是會在第一個請求時才算（實測地端
    #   47 秒）。預熱名不副實，而且從外面看不出來。
    for side in settings().warm_sides():
        deps.bot_for(side).retriever.doc_vecs()
    yield


app = FastAPI(
    title="RAG Chat-bot API",
    version="0.1.0",
    summary="媒合會職缺查詢。雲端（Gemini）與地端（llama.cpp）同一組端點。",
    lifespan=lifespan,
)


# 瀏覽器前端要的。Stage 3 沒加是因為當時只有 Streamlit（伺服器端呼叫，
# 不受 CORS 管）——為了還不存在的客戶端先加中介層是多餘的。現在它存在了。
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings().cors_list(),
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.get("/health", response_model=Health)
def health(registry: RegistryDep):
    return Health(status="ok", sides=list(providers.SIDES), loaded=deps.loaded(),
                  chunks=deps.chunk_counts(), documents=len(registry.list()))


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(registry: RegistryDep):
    """列出**上傳的**文件。建置階段的基礎語料不在這裡，也刪不掉。"""
    return [DocumentOut(**row) for row in registry.list()]


@app.post("/documents", status_code=201, response_model=UploadResult)
async def upload_document(file: UploadFile, registry: RegistryDep):
    """上傳一份職缺 PDF，當場解析、切塊、進索引。

    同一個檔名視為同一份文件：內容變了就是更新（舊的塊會被換掉），
    內容一樣就回 409——沒有事情要做。
    """
    cfg = settings()
    name = file.filename or ""
    suffix = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
    if suffix not in cfg.upload_suffixes():
        raise HTTPException(400, "只收 %s；只吃媒合會職缺表那個格式，理由見 "
                                 "shared/ingest.py" % "、".join(cfg.upload_suffixes()))

    data = await file.read()
    if len(data) > cfg.max_upload_bytes:
        raise HTTPException(413, "檔案超過 %d MB"
                                 % (cfg.max_upload_bytes // 1024 // 1024))

    doc_id, version = doc_id_for(name), version_of(data)
    existing = registry.get(doc_id)
    if existing and existing["version"] == version:
        raise HTTPException(409, f"「{name}」已經上傳過同樣內容的版本了")

    try:
        chunks, jobs = docs_from_pdf(data, source=doc_id)
    except Exception as e:
        raise HTTPException(400, f"這份 PDF 解析不了：{e}") from e
    if not chunks:
        raise HTTPException(400, "解析不出任何職缺——確認這是媒合會職缺表的格式")

    registry.put(name, file.content_type or "application/pdf", len(data), version,
                 chunks, jobs)
    return UploadResult(
        document=DocumentOut(**registry.list_one(doc_id)),
        index=deps.refresh_loaded(),
    )


@app.delete("/documents/{doc_id}", response_model=DeleteResult)
def delete_document(doc_id: str, registry: RegistryDep):
    if not registry.delete(doc_id):
        raise HTTPException(404, f"沒有這份文件：{doc_id}")
    return DeleteResult(doc_id=doc_id, index=deps.refresh_loaded())


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, factory: BotFactoryDep):
    """檢索 + 生成。

    history 進來什麼樣、回去就是更新後的樣子——伺服器不記任何東西，
    理由見 api/schemas.py 開頭。
    """
    bot, history = _prepare(req, factory)
    try:
        reply, hits, usage = bot.ask(req.question, history, k=req.k)
    except Exception as e:
        raise _as_http(bot, e) from e

    return ChatResponse(
        reply=reply,
        sources=[_source(d, s) for d, s in hits],
        usage=Usage(prompt=usage.prompt, output=usage.output),
        history=[Message(**m) for m in history],
    )


def _prepare(req, factory):
    """兩個端點共用：挑出那一邊的 bot，並把 history 轉成 ask() 吃的形狀。"""
    try:
        bot = factory(req.side)
    except Exception as e:      # 例如雲端沒設金鑰
        raise HTTPException(503, f"{req.side} 這一邊起不來：{e}") from e
    return bot, [m.model_dump() for m in req.history]    # ask() 會就地更新這個 list


def _source(doc, score):
    return Source(code=doc.group, label=doc.label, score=score)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, factory: BotFactoryDep):
    """跟 /chat 一樣，但用 SSE 逐段吐字。

    事件有四種：
        sources  檢索結果。**第一個送出**——檢索比生成快得多，UI 可以先把
                 來源列出來，不必等模型講完
        token    一段文字
        done     usage 與更新後的 history
        error    生成中途壞掉。這時 HTTP 狀態已經送出去了（200），改不了，
                 所以只能用事件回報——呼叫端一定要處理這個事件

    連線階段的錯誤仍然是正常的 HTTP 狀態碼：下面會先「預抽」第一段，
    抽得出來才開始串流。代價是 sources 要等到模型吐第一個字才送得出去
    （實測 0.2 秒），換到的是 503/429 不會偽裝成 200。
    """
    bot, history = _prepare(req, factory)

    try:
        turn = bot.ask_stream(req.question, history, k=req.k)
        pieces = iter(turn)
        first = next(pieces, None)          # ← 預抽，讓連線期的錯誤變成正確狀態碼
    except Exception as e:
        raise _as_http(bot, e) from e

    def events():
        # .model_dump()：這條路是我們自己呼叫 json.dumps，沒有 FastAPI 幫忙
        # 序列化 pydantic 模型。
        yield _sse("sources", {"sources": [_source(d, s).model_dump()
                                          for d, s in turn.hits]})
        try:
            for piece in ([first] if first is not None else []):
                yield _sse("token", {"text": piece})
            for piece in pieces:
                yield _sse("token", {"text": piece})
        except Exception as e:
            yield _sse("error", {"detail": bot.llm.explain(e) or str(e)})
            return
        yield _sse("done", {"usage": {"prompt": turn.usage.prompt,
                                      "output": turn.usage.output},
                            "history": history})

    # ⚠ charset 一定要宣告。SSE 的 text/event-stream 若沒寫 charset，HTTP 對
    #   text/* 的規定是退回 ISO-8859-1——實測 requests 就是這樣猜，中文會變成
    #   'å¥½ç' 這種東西，而且完全不報錯。
    return StreamingResponse(events(), media_type="text/event-stream; charset=utf-8")


def _sse(event, data):
    """SSE 的一則事件。data 不能有裸換行——JSON 編碼剛好把換行變成 \n。"""
    return "event: %s\ndata: %s\n\n" % (
        event, json.dumps(data, ensure_ascii=False))


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
