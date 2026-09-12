# RAG Chat-bot 實作

一組最小可跑的 RAG（Retrieval-Augmented Generation）聊天機器人練習，同一套流程各做了**雲端版**與**地端版**兩種實作，方便對照兩者的差異。

知識庫是 11 家公司的媒合會職缺表（PDF），共 30 個職缺。

## 架構

RAG 拆成兩段：先用 embedding 從知識庫撈出最相關的段落，再把段落和問題一起丟給 LLM 回答。

雲端與地端並排成兩個資料夾，模組**同名同形狀**——換邊只要換 `sys.path` 那一行：

```
├── cloud_app.py            Streamlit 介面（雲端版）
├── onperm_app.py           Streamlit 介面（地端版）
├── data/
│   └── jobs.json           知識庫（由 tools/build_jobs.py 產生）
├── shared/
│   └── knowledge.py        Doc 結構 + SYSTEM，兩邊共用
├── cloud/
│   ├── rag.py              Gemini gemini-embedding-001，輸出維度截短成 768
│   └── chat_bot.py         Gemini gemini-flash-latest，含多輪記憶與 token 計量
├── onperm/
│   ├── rag.py              sentence-transformers + intfloat/multilingual-e5-small
│   └── chat_bot.py         走 llama.cpp server 的 OpenAI 相容 API
└── tools/
    ├── build_jobs.py       職缺 PDF → data/jobs.json
    └── probe_threshold.py  量 retrieve 的 min_score 該設多少
```

### Doc 的兩個長度上限

`shared/knowledge.py` 的 `Doc` 把 `text` 和 `full` 分成兩個欄位，因為這兩個上限是不同的東西：

| 欄位 | 上限 | 受誰限制 |
|------|------|---------|
| `text` | **512 token** | embedding 模型（e5-small）。超過會被**靜默截斷**，不報錯，尾巴永遠檢索不到 |
| `full` | 4096~8192 token | 生成模型的 context |

所以一個職缺會被切成好幾塊各自 embed，但撈到任一塊都餵完整職缺。以整份職缺去 embed 的話，30 個裡有 13 個（42%）超過 512、最長的 1504 token。

### 看起來像、但不能合併的地方

`ask()` 雲端回傳 `(reply, hits, usage)`、地端只有 `(reply, hits)`（本機推論不計費）；history 格式一邊是 Gemini 的 `parts`/`model`、一邊是 OpenAI 的 `content`/`assistant`。

## 安裝

```bash
pip install -r requirements.txt
```

## 建知識庫

```bash
python tools/build_jobs.py "<職缺 PDF 資料夾>"
```

產出 `data/jobs.json`：30 個職缺 → 140 塊。資料夾結構是每家公司一個子資料夾、裡面一份 PDF。

> ⚠ 重建 `jobs.json` 之後要重跑 `tools/probe_threshold.py`。兩邊的 `min_score` 都是對「這批」資料量出來的，不是通用常數。

## 雲端版（Gemini）

1. 到 [Google AI Studio](https://aistudio.google.com/apikey) 申請 API key
2. 複製 `.env.example` 成 `.env` 並填入金鑰
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
2. 下載一個 GGUF 模型，例如 `Qwen2.5-3B-Instruct-Q4_K_M.gguf`（約 1.9GB）。
3. 啟動 server：

```bash
llama-server.exe -m <模型路徑>.gguf --port 8080 -c 8192 -ngl 99 --device Vulkan0
```

`-c 8192` 是必要的：`k=5` 撈回五個完整職缺約 3700 token，`-c 4096` 會沒有空間留給生成，server 直接回 400。

`--device` 的編號先用 `llama-server.exe --list-devices` 確認——如果機器上還有內顯，不指定的話模型可能被拆到內顯上，會慢很多。

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

### min_score 已經不是離題守門員了

這批資料量出來的間隙只有 **0.003** 寬（FAQ 時代是 0.046）：離題問句「推薦一家餐廳」0.863，正常問句「耐能智慧在徵什麼人」0.866，幾乎貼在一起。

原因是統計性的，不是門檻選錯：候選塊從 5 個變成 140 個，**雜訊的最大值被推高，但正確答案的分數不會跟著漲**。同一個離題問句在 5 塊裡最高分 0.837，在 140 塊裡是 0.863。

試過的替代方案都沒用：改成 z 分數（相對於該次分佈）間隙變成 **-0.41**，兩群直接重疊；改成字元 bigram 的 BM25 間隙 **-0.31**，`Kubernetes` 只拿 3.87 輸給「推薦一家餐廳」的 4.18。

所以 `min_score` 現在只當下限用，擋掉明顯無關的塊；**離題的判斷交給模型**，靠 `SYSTEM` 裡那句「資料沒提到的就說『資料裡沒有』」。實測三個離題問句都被正確拒答。

## 說明

`.env` 已列入 `.gitignore`，金鑰不會進版控。模型檔（`*.gguf`、`models/`）也不進版控。
