# RAG Chat-bot 實作

一組最小可跑的 RAG（Retrieval-Augmented Generation）聊天機器人練習，同一套流程各做了**雲端版**與**地端版**兩種實作，方便對照兩者的差異。

## 架構

RAG 拆成兩段：先用 embedding 從知識庫撈出最相關的段落，再把段落和問題一起丟給 LLM 回答。

雲端與地端並排成兩個資料夾，模組**同名同形狀**——換邊只要換 `sys.path` 那一行：

```
├── cloud_app.py            Streamlit 介面（雲端版）
├── onperm_app.py           Streamlit 介面（地端版）
├── shared/
│   └── knowledge.py        DOCS + SYSTEM，兩邊唯一共用的東西
├── cloud/
│   ├── rag.py              Gemini gemini-embedding-001，輸出維度截短成 768
│   └── chat_bot.py         Gemini gemini-flash-latest，含多輪記憶與 token 計量
├── onperm/
│   ├── rag.py              sentence-transformers + intfloat/multilingual-e5-small
│   └── chat_bot.py         走 llama.cpp server 的 OpenAI 相容 API
└── tools/
    └── probe_threshold.py  量 retrieve 的 min_score 該設多少
```

看起來像、但**不能合併**的地方：`ask()` 雲端回傳 `(reply, hits, usage)`、地端只有 `(reply, hits)`（本機推論不計費）；history 格式一邊是 Gemini 的 `parts`/`model`、一邊是 OpenAI 的 `content`/`assistant`。

知識庫是 `shared/knowledge.py` 裡寫死的 5 段電商 FAQ（退款、運費、客服時間、會員等級、保固），改 `DOCS` 就能換內容。

> ⚠ 兩邊的 `min_score`（雲端 0.70、地端 0.84）是**各自對這批資料量出來的**，不是通用常數。改了 `DOCS` 或換 embedding 模型，要重跑 `tools/probe_threshold.py` 重量。

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
python cloud/rag.py           # 只測檢索
python cloud/chat_bot.py      # 互動對話（CLI）
streamlit run cloud_app.py    # 網頁介面
```

免費方案有用量上限（觀測到每天約 20 次生成），超過會收到 429。

## 地端版（llama.cpp）

需要先在本機把 llama.cpp server 跑起來，聽在 `http://localhost:8080`。不需要 Docker、不需要 WSL：

1. 到 [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) 下載 Windows 版 binary。注意「latest」那個 tag 裡沒有執行檔，要找 `bXXXXX` 的每日 build。
   - **Vulkan 版**（`llama-bXXXXX-bin-win-vulkan-x64.zip`，約 32MB）最省事——runtime 由顯卡驅動自帶，不用另外抓 CUDA 的 400MB cudart，而且 Pascal / Turing 通吃。
   - CUDA 版要另外抓同一頁的 `cudart-*.zip`，並確認驅動版本對得上（CUDA 13 已不支援 Pascal）。
2. 下載一個 GGUF 模型，例如 `Qwen2.5-3B-Instruct-Q4_K_M.gguf`（約 1.9GB，4GB 顯存放得下）。
3. 啟動 server：

```bash
llama-server.exe -m <模型路徑>.gguf --port 8080 -c 4096 -ngl 99 --device Vulkan0
```

`--device` 後面的編號先用 `llama-server.exe --list-devices` 確認——如果機器上還有內顯，不指定的話模型可能被拆到內顯上，會慢很多。

4. 另開一個終端執行：

```bash
python onperm/rag.py          # 只測檢索（不需要 server）
python onperm/chat_bot.py     # 互動對話（CLI，需要 server）
streamlit run onperm_app.py   # 網頁介面（需要 server）
```

第一次跑地端檢索會自動下載 e5-small 模型（約 470MB）。

## 量門檻

```bash
python tools/probe_threshold.py onperm    # 純本機，不花錢
python tools/probe_threshold.py cloud     # 會打 embedding API
```

會列出「資料裡有的 / 沒有的 / 跟隨問句」三組問題的相似度分佈，算出兩群之間的間隙並建議門檻。

## 說明

`.env` 已列入 `.gitignore`，金鑰不會進版控。模型檔（`*.gguf`、`models/`）也不進版控。
