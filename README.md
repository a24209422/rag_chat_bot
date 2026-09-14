# RAG Chat-bot 實作

一組最小可跑的 RAG（Retrieval-Augmented Generation）聊天機器人練習，同一套流程各做了**雲端版**與**地端版**兩種實作，方便對照兩者的差異。

知識庫是 11 家公司的媒合會職缺表（PDF），共 30 個職缺。

> 設計取捨、量出來的數字、踩過的坑，都在 [docs/design-notes.md](docs/design-notes.md)。
> 這份只寫怎麼跑。

## 示範

https://github.com/user-attachments/assets/74c546b2-26bd-4cd9-bbd9-9828e806a3e9

問「有哪些台北的職缺？」。來源列比模型的字先到——檢索比生成快得多。吐完字展開來源列：
**模型的散文列 15 筆，來源列 20 筆**，後者才是完整的。理由見
[完整的清單由程式給，不是模型](docs/design-notes.md#完整的清單由程式給不是模型)。

## 架構

RAG 拆成兩段：先用 embedding 從知識庫撈出最相關的段落，再把段落和問題一起丟給 LLM 回答。

雲端與地端並排成兩個資料夾，共用的部分全在 `shared/`。兩邊的檢索器繼承同一個 base
class，換邊只要換 `providers.chat_for()` 的參數：

```
├── providers.py            retriever_for() / llm_for() / chat_for()：依名稱組零件
├── api/                    ← FastAPI 後端
│   ├── main.py             /health、/chat、/chat/stream、/documents
│   ├── schemas.py          請求與回應的形狀（pydantic）
│   └── deps.py             side → ChatBot，第一次用到才建
├── frontend/               ← React + Vite 前端（見 frontend/README.md）
├── cloud_app.py            Streamlit 雲端版進入點（畫面在 shared/app.py）
├── onperm_app.py           Streamlit 地端版進入點
├── data/
│   ├── jobs.json           基礎語料（由 tools.build_jobs 產生）
│   ├── registry.db         上傳文件的登記簿（執行期產生，不進版控）
│   └── store_*.npz         各邊的向量索引（執行期產生，不進版控）
├── shared/                 ← 兩邊共用的，全部在這裡
│   ├── settings.py         所有會因環境而異的設定（讀 .env）
│   ├── knowledge.py        Doc 結構 + load_docs() + SYSTEM
│   ├── facets.py           把自由文字的 metadata 正規化成可精確比對的分類
│   ├── ingest.py           職缺 PDF → 塊（建置與上傳共用同一份解析）
│   ├── store.py            向量儲存層：增刪、持久化
│   ├── registry.py         上傳文件的登記簿（SQLite）
│   ├── retriever.py        BaseRetriever：檢索邏輯，子類只補 embed()
│   ├── llm.py              BaseLLM：跟模型講話的介面，子類補線路格式
│   ├── chat_bot.py         ChatBot：檢索 + 生成的唯一一份實作
│   ├── api_client.py       打後端 API 的客戶端（Streamlit 與 CLI 用）
│   └── app.py              Streamlit 畫面的唯一一份實作
├── cloud/                  ← 只剩「Gemini 特有的事」
│   ├── client.py           Gemini client 的唯一建構點（讀金鑰只有這一處）
│   ├── rag.py              CloudRetriever：gemini-embedding-001，維度截短成 768
│   └── llm.py              GeminiLLM：contents/parts 格式、思考 token、錯誤翻譯
├── onperm/                 ← 只剩「llama.cpp 特有的事」
│   ├── rag.py              OnpremRetriever：intfloat/multilingual-e5-small
│   └── llm.py              LlamaCppLLM：OpenAI 相容格式、usage、錯誤翻譯
└── tools/
    ├── build_jobs.py       職缺 PDF → data/jobs.json
    ├── probe_threshold.py  量 retrieve 的 min_score 該設多少
    └── chat.py             互動式對話（CLI，兩邊共用）
```

設定（模型名、server 位址、逾時、溫度、`k`、history 上限）全部在 `shared/settings.py`，
可用 `.env` 或環境變數覆寫，項目見 `.env.example`。

## 安裝

```bash
pip install -r requirements.txt
```

版本是鎖死的（`==`），鎖在「整批測試都通過」的那一組。要改開發相關的東西再裝
`requirements-dev.txt`。

## 建知識庫

```bash
python -m tools.build_jobs "<職缺 PDF 資料夾>"
```

產出 `data/jobs.json`：30 個職缺 → 140 塊。資料夾結構是每家公司一個子資料夾、裡面一份 PDF。

> ⚠ 重建 `jobs.json` 之後要重跑 `python -m tools.probe_threshold <side>`。兩邊的
> `min_score` 都是對「這批」資料量出來的，不是通用常數——
> [怎麼量的](docs/design-notes.md#檢索門檻)。

## 一鍵啟動

```powershell
.\start.ps1            # 後端 + 前端，雲端（Gemini）可用
.\start.ps1 -Local     # 再帶上 llama.cpp server，地端推論才跑得動
.\start.ps1 -Stop      # 通通關掉
```

雙擊 `start.bat` 等同第一個。腳本沒有做任何新的事，只是把下面的手動指令綁在一起，
外加三件手動時容易忘的：埠已經有人在聽就沿用、`-Local` 會設 `API_WARM=onperm` 預熱
（第一個請求 47.6 秒 → 2.6 秒）、不加 `--reload`（它的子行程關不乾淨）。

`-Local` 要先在 `.env` 補 llama.cpp 的執行檔與模型路徑（格式見 `.env.example`）。

> ⚠ `start.ps1` 必須存成 **UTF-8 with BOM**。Windows PowerShell 5.1 讀 `.ps1` 用
> 系統 ANSI（這台機器是 cp950），沒有 BOM 的話中文全部變亂碼。編輯器另存新檔時很容易
> 把它弄掉，而症狀跟程式邏輯無關，會找很久。

## 雲端版（Gemini）

1. 到 [Google AI Studio](https://aistudio.google.com/apikey) 申請 API key
2. 複製 `.env.example` 成 `.env` 並填入金鑰
3. 執行：

```bash
python -m cloud.rag             # 只測檢索
python -m tools.chat cloud      # 互動對話（CLI，直接呼叫，不需要後端）
uvicorn api.main:app --reload   # 後端 API
streamlit run cloud_app.py      # 網頁介面（另開一個終端，需要後端）
```

免費方案有用量上限（觀測到每天約 20 次生成），超過會收到 429。建索引還會撞到每分鐘的
request 配額，所以雲端這側有向量快取——見
[雲端的兩個配額陷阱](docs/design-notes.md#雲端的兩個配額陷阱)。

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

`-c 8192` 是必要的：`k=5` 撈回五個完整職缺約 3700 token，`-c 4096` 會沒有空間留給生成，
server 直接回 400。`--device` 的編號先用 `llama-server.exe --list-devices` 確認——
機器上還有內顯的話，不指定可能被拆過去，會慢很多。

4. 另開一個終端執行：

```bash
python -m onperm.rag            # 只測檢索（不需要 server）
python -m tools.chat onperm     # 互動對話（CLI，直接呼叫，不需要後端）
uvicorn api.main:app --reload   # 後端 API
streamlit run onperm_app.py     # 網頁介面（另開一個終端，需要後端）
```

第一次跑地端檢索會自動下載 e5-small 模型（約 470MB）。

## 後端 API

問答邏輯包成 HTTP 服務，兩個前端打同一組 API：

```bash
uvicorn api.main:app --reload               # 後端，聽 http://localhost:8000

cd frontend && npm install && npm run dev   # React，http://localhost:5173
streamlit run cloud_app.py                  # 或 Streamlit
```

互動式文件在 <http://localhost:8000/docs>（FastAPI 從 `api/schemas.py` 自動生成）。

| 方法 | 路徑 | 說明 |
|---|---|---|
| GET | `/health` | 健康檢查：哪幾邊的索引算好了、各有幾塊、上傳了幾份文件 |
| POST | `/chat` | 檢索 + 生成，一次回完 |
| POST | `/chat/stream` | 同上，但用 SSE 逐段吐字（`sources` / `token` / `done` / `error`） |
| GET | `/documents` | 列出上傳的文件 |
| POST | `/documents` | 上傳一份職缺 PDF，當場切塊進索引 |
| DELETE | `/documents/{id}` | 刪除文件與它的塊 |

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"question":"有哪些無人機相關的職缺？","side":"onperm"}'
```

```json
{
  "reply": "資料中有以下無人機相關的職缺…",
  "sources": [{"code": "MAT-04", "label": "MAT-04 · 神耀科技 · 無人機硬體工程師 · 臺北市內湖區", "score": 0.906}],
  "usage": {"prompt": 2764, "output": 78},
  "history": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]
}
```

`history` 由呼叫端帶進來、原樣帶回去，伺服器不存——
[為什麼](docs/design-notes.md#後端不記-history)。串流的事件順序、錯誤碼、索引什麼時候建，
見 [後端 API](docs/design-notes.md#後端-api)。

## 前端

`frontend/` 是 React + Vite + TypeScript，功能：串流對話、來源列（代號獨立顯示）、
雲端／地端切換、token 計量、文件上傳與刪除。細節見 [frontend/README.md](frontend/README.md)。

## 動態知識庫

執行期可以上傳職缺 PDF，當場解析、切塊、進索引；也可以刪掉。同一個檔名視為同一份文件，
內容變了就是更新，內容一樣回 409。

```bash
curl -F "file=@新公司職缺.pdf" http://localhost:8000/documents
curl http://localhost:8000/documents
curl -X DELETE http://localhost:8000/documents/<doc_id>
```

索引是增量同步的（上傳一份文件 0.08 秒，從零重建地端要 49.7 秒），細節與已知限制見
[動態知識庫](docs/design-notes.md#動態知識庫)。

## 測試與 lint

```bash
python -m pytest          # 整批約 3 秒
python -m ruff check .    # lint
pre-commit install        # 裝一次，之後每次 commit 自動跑上面兩項
```

測試刻意不碰模型也不碰網路，所以整批跑完是秒級的。各檔案測什麼、怎麼避開外部依賴，
見 [測試](docs/design-notes.md#測試)。

## 說明

`.env` 已列入 `.gitignore`，金鑰不會進版控。模型檔（`*.gguf`、`models/`）也不進版控。
