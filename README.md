# RAG Chat-bot 實作

一組最小可跑的 RAG（Retrieval-Augmented Generation）聊天機器人練習，同一套流程各做了**雲端版**與**地端版**兩種實作，方便對照兩者的差異。

## 架構

RAG 拆成兩段：先用 embedding 從知識庫撈出最相關的段落，再把段落和問題一起丟給 LLM 回答。

| 檔案 | 角色 | 說明 |
|------|------|------|
| [`scr/cloud_rag.py`](scr/cloud_rag.py) | 雲端檢索 | Gemini `gemini-embedding-001`，輸出維度截短成 768 |
| [`scr/cloud_chat_bot.py`](scr/cloud_chat_bot.py) | 雲端對話 | Gemini `gemini-flash-latest`，含多輪記憶 |
| [`scr/OnPerm_rag.py`](scr/OnPerm_rag.py) | 地端檢索 | `sentence-transformers` + `intfloat/multilingual-e5-small` |
| [`scr/OnPerm_chat_bot.py`](scr/OnPerm_chat_bot.py) | 地端對話 | 走 llama.cpp server 的 OpenAI 相容 API |

知識庫目前是寫死在程式裡的 5 段電商 FAQ（退款、運費、客服時間、會員等級、保固），改 `DOCS` 就能換內容。

## 安裝

```bash
pip install -r requirements.txt
```

## 雲端版（Gemini）

1. 到 [Google AI Studio](https://aistudio.google.com/apikey) 申請 API key
2. 複製 `.env.example` 成 `.env`，填入自己的金鑰：

```bash
cp .env.example .env
```

3. 執行：

```bash
python scr/cloud_rag.py        # 只測檢索
python scr/cloud_chat_bot.py   # 互動對話
```

## 地端版（llama.cpp）

需要先在本機把 llama.cpp server 跑起來，聽在 `http://localhost:8080`：

```bash
python scr/OnPerm_rag.py        # 只測檢索
python scr/OnPerm_chat_bot.py   # 互動對話
```

第一次跑地端檢索會自動下載 e5-small 模型（約 470MB）。

## 說明

`.env` 已列入 `.gitignore`，金鑰不會進版控。
