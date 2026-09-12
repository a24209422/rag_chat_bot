# RAG Chat-bot 實作

一組最小可跑的 RAG（Retrieval-Augmented Generation）聊天機器人練習，同一套流程各做了**雲端版**與**地端版**兩種實作，方便對照兩者的差異。

知識庫是 11 家公司的媒合會職缺表（PDF），共 30 個職缺。

## 架構

RAG 拆成兩段：先用 embedding 從知識庫撈出最相關的段落，再把段落和問題一起丟給 LLM 回答。

雲端與地端並排成兩個資料夾。兩邊的檢索器繼承同一個 base class，所以「同名同形狀」是
繼承保證的，不是靠約定；換邊只要換 `providers.chat_for()` 的參數：

```
├── providers.py            retriever_for() / llm_for() / chat_for()：依名稱組零件
├── api/                    ← FastAPI 後端
│   ├── main.py             GET /health、POST /chat
│   ├── schemas.py          請求與回應的形狀（pydantic）
│   └── deps.py             side → ChatBot，第一次用到才建
├── cloud_app.py            雲端版進入點（幾行，畫面在 shared/app.py）
├── onperm_app.py           地端版進入點
├── data/
│   └── jobs.json           知識庫（由 tools.build_jobs 產生）
├── shared/                 ← 兩邊共用的，全部在這裡
│   ├── settings.py         所有會因環境而異的設定（讀 .env）
│   ├── knowledge.py        Doc 結構 + load_docs() + SYSTEM
│   ├── facets.py           把自由文字的 metadata 正規化成可精確比對的分類
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

### Doc 的兩個長度上限

`shared/knowledge.py` 的 `Doc` 把 `text` 和 `full` 分成兩個欄位，因為這兩個上限是不同的東西：

| 欄位 | 上限 | 受誰限制 |
|------|------|---------|
| `text` | **512 token** | embedding 模型（e5-small）。超過會被**靜默截斷**，不報錯，尾巴永遠檢索不到 |
| `full` | 4096~8192 token | 生成模型的 context |

所以一個職缺會被切成好幾塊各自 embed，但撈到任一塊都餵完整職缺。以整份職缺去 embed 的話，30 個裡有 13 個（42%）超過 512、最長的 1504 token。

### 「不能合併」是錯的，後來推翻了

這個專案曾經有 `CloudChatBot` 和 `OnpremChatBot` 兩份幾乎一樣的實作，README 上寫著「刻意不合併」，理由有兩個：

1. history 格式不同（Gemini 的 `parts`/`model` vs OpenAI 的 `content`/`assistant`）
2. `ask()` 回傳不同——「地端本機推論不計費，所以沒有 usage」

兩個理由後來都不成立：

1. **那是線路格式，屬於傳輸層。** 讓它決定上層架構是搞錯分層——把轉換收進各自的 LLM client，上層只認一種中性格式就好。
2. **「不計費」不等於「量不到」。** llama.cpp 的 `/v1/chat/completions` 實測會回 `prompt_tokens` 與 `completion_tokens`。而且地端**更需要**這個數字：它告訴你離 `-c` 的上限還有多遠（開 `-c 4096` 撈五個職缺會直接回 400，就是這個坑）。

所以現在 `ChatBot` 只有一份，兩邊的 `ask()` 都回 `(reply, hits, usage)`。差異縮到三個可替換的零件：

| 零件 | 雲端 | 地端 |
|------|------|------|
| `Retriever` | Gemini embedding，`min_score=0.65` | e5-small，`min_score=0.82` |
| `LLM` | `contents`/`parts`、思考 token | OpenAI 相容、`usage` 欄位 |
| history 上限 | 不砍（context 寬裕） | 11 則 |

連 Streamlit 畫面也收成一份了（`shared/app.py`），因為 history 統一之後兩支 app 只差標題。

錯誤訊息的翻譯也歸各自的 LLM client：`explain(exc)` 把「這家可以預期會發生的錯」翻成人話（Gemini 的 503/429、llama.cpp 的連不上/逾時），不認得就回 `None`。UI 只要寫 `llm.explain(e) or str(e)`——429 對 Gemini 是配額，對別家可能是別的意思，那是 provider 的知識。

### 設定集中在一處

模型名、server 位址、逾時、溫度、`k`、history 上限，全部在 `shared/settings.py`（pydantic-settings，讀 `.env`），可用環境變數覆寫。可改的項目見 `.env.example`。

> ⚠ **`min_score` 刻意不在設定裡。** 它不是旋鈕，是 `tools/probe_threshold.py` 對「這個 embedding 模型 ＋ 這批語料」量出來的結果，換任一邊都要重量。放進 `.env` 會讓它看起來像可以隨手調的參數，那正是最危險的誤解。要臨時試別的值就傳建構子參數：`retriever_for("cloud", min_score=0.7)`。

### 狀態都掛在實例上，沒有模組層全域

`Retriever` 的索引、區名詞彙表、知識庫，以及 Gemini client，全部是實例屬性而且延遲建立——建構子不讀檔、不讀金鑰、不載模型。這帶來幾件事：

- 沒有 `data/jobs.json` 或 `GEMINI_API_KEY` 也 import 得起來（測試與 CI 需要）
- 同一個 process 裡可以同時存在雲端與地端兩個 retriever
- 測試可以傳自己的知識庫：`retriever_for("onperm", docs=[Doc(...)])`

> ⚠ 舊版靠 `sys.path.insert` 切換兩邊，而 `cloud/rag.py` 與 `onperm/rag.py` 都叫 `rag`。`sys.modules` 會快取，所以一個 process 裡**只載得進其中一邊**，第二次 `import rag` 會靜默拿到第一邊——不報錯。Streamlit 一次只跑一支所以碰不到，但測試和後端服務都會撞上。

## 安裝

```bash
pip install -r requirements.txt
```

版本是鎖死的（`==`），鎖在「99 個測試都通過」的那一組。要改開發相關的東西就裝：

```bash
pip install -r requirements-dev.txt
```

## 建知識庫

```bash
python -m tools.build_jobs "<職缺 PDF 資料夾>"
```

產出 `data/jobs.json`：30 個職缺 → 140 塊。資料夾結構是每家公司一個子資料夾、裡面一份 PDF。

> ⚠ 重建 `jobs.json` 之後要重跑 `tools/probe_threshold.py`。兩邊的 `min_score` 都是對「這批」資料量出來的，不是通用常數。

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
python -m onperm.rag            # 只測檢索（不需要 server）
python -m tools.chat onperm     # 互動對話（CLI，直接呼叫，不需要後端）
uvicorn api.main:app --reload   # 後端 API
streamlit run onperm_app.py     # 網頁介面（另開一個終端，需要後端）
```

第一次跑地端檢索會自動下載 e5-small 模型（約 470MB）。

## 量門檻

```bash
python -m tools.probe_threshold onperm    # 純本機，不花錢
python -m tools.probe_threshold cloud     # 會打 embedding API
```

會列出「資料裡有的 / 沒有的 / 跟隨問句」三組問題的相似度分佈，算出兩群之間的間隙，判斷這個間隙可不可信，再決定要給門檻還是給下限。

判斷可靠度用的是**穩定性檢查**：逐一抽掉每個問句重算間隙。如果間隙是由單一個問句撐著的，抽掉它就會大幅改變——那個數字就不能信。這比拍一個固定的比值門檻誠實，而且直接回答「再多一個測試問句會不會翻盤」。

| | 穩定性 | 判定 | 建議 |
|---|---|---|---|
| 地端 e5 | 抽掉一個問句間隙變 **7.7 倍** | 不可用 | 下限 `0.82` |
| 雲端 Gemini | **1.4 倍** | 可用 | 門檻 `0.65` |

判定「不可用」時不會硬給一個門檻，而是說明該改用什麼（交給模型 / metadata 過濾），並給一個寬鬆下限（`有的最低 − 0.05`）——那是用來擋掉明顯無關的塊，不是擋問句。

> 每組只有 5 個問句，所有判定都是粗估。輸出裡有標註。

### 雲端與地端的門檻差 30 倍

同樣 140 塊、同樣 15 個問句，兩邊量出來的間隙差很多：

| | 間隙 | 離題分數 | 正常分數 | 門檻 |
|---|---|---|---|---|
| 地端 `multilingual-e5-small` | **0.003** | 0.819 ~ 0.863 | 0.866 ~ 0.908 | 0.82（只當下限） |
| 雲端 `gemini-embedding-001` | **0.092** | 0.556 ~ 0.604 | 0.695 ~ 0.795 | 0.65（真的能擋） |

所以間隙收窄**不是**單純候選數變多造成的：e5 把所有分數擠在 0.75~0.91 的窄帶裡，Gemini 的分佈散得多。實際差異看得出來——問「推薦一家餐廳」，雲端撈回 **0 筆**（門檻擋住，直接短路不打生成 API），地端撈回 5 筆（得靠模型拒答）。

另一個差異：Gemini 會照 `SYSTEM` 的指示在答案裡附上代號，本機 Qwen2.5-3B 不管怎麼調 prompt 和溫度都不肯抄。

雲端的 `min_score` 原本是 FAQ 時代的 0.70，換成職缺資料後會擋掉「有沒有可以全遠端的工作？」——它只有 0.695。重量後改成 0.65。

### 地端的 min_score 已經不是離題守門員了

這批資料量出來的間隙只有 **0.003** 寬（FAQ 時代是 0.046）：離題問句「推薦一家餐廳」0.863，正常問句「耐能智慧在徵什麼人」0.866，幾乎貼在一起。

原因是統計性的，不是門檻選錯：候選塊從 5 個變成 140 個，**雜訊的最大值被推高，但正確答案的分數不會跟著漲**。同一個離題問句在 5 塊裡最高分 0.837，在 140 塊裡是 0.863。

試過的替代方案都沒用：改成 z 分數（相對於該次分佈）間隙變成 **-0.41**，兩群直接重疊；改成字元 bigram 的 BM25 間隙 **-0.31**，`Kubernetes` 只拿 3.87 輸給「推薦一家餐廳」的 4.18。

所以 `min_score` 現在只當下限用，擋掉明顯無關的塊；**離題的判斷交給模型**，靠 `SYSTEM` 裡那句「資料沒提到的就說『資料裡沒有』」。實測三個離題問句都被正確拒答。

## metadata 過濾

純向量檢索答不好「篩選型」問題：問「有哪些台北的職缺」時，語意相似度只會給你最像的前 k 個，沒辦法保證「全部」。但地點、工作性質這些欄位本來就是結構化的，直接篩就能保證完整。

`shared/facets.py` 在建置階段把自由文字正規化成乾淨的分類，查詢時才能精確比對：

| 問句 | 抽出的條件 | 撈回 |
|------|-----------|------|
| 有哪些台北的職缺？ | `city=台北` | **20 個（全部）** |
| 內湖的職缺有哪些？ | `district=內湖` | **8 個（全部）** |
| 三重的實習 | `district=三重, kind=實習` | 2 個 |
| 新北市有什麼工作？ | `city=新北` | 4 個 |
| 有沒有可以遠端的？ | `remote=True` | 4 個 |
| 有哪些實習機會？ | `kind=實習` | 7 個 |
| 台北的實習有哪些？ | `city=台北, kind=實習` | 3 個 |
| 有哪些無人機相關的職缺？ | （無）→ 純向量檢索 | 5 個 |

有過濾條件時會放掉 `min_score` 和 `k`——篩選已經保證相關，分數門檻反而會誤殺；`k` 不放開的話問「台北的職缺」只給 5 個。

### 正規化踩到的三個坑

原始值很髒，不能直接當分類用：

- **台／臺兩種寫法都有**：21 個寫「台北」、4 個寫「臺北」。只比對一種會漏掉 4 個。
- **有公司沒填地點**，留著範本文字「【待填：公司地點，例：台北市內湖區】」。不砍掉的話會被裡面的「例：台北市內湖區」誤判成台北。
- **括號裡是地標不是城市**：「新北市三重區（捷運台北橋站步行約 30 秒）」的「台北橋」是站名。判斷城市前要砍掉括號，不然這三個職缺會同時被判成台北和新北。

### 區名是抽取的，不是硬編的

城市用固定的對照表（就那幾個），但**區名用正規表示式從資料抽**，新資料進來才能自動涵蓋。抽出來的詞彙表目前是：三重、中和、中山、中正、內湖、大安、永康、萬華。

錨點很重要：區名必須緊跟在「市／縣」或城市名之後，否則「台北大安區」會被抓成「北大安」。

查詢側沒辦法用同一個正規表示式——問句裡的「內湖」沒有「區」字可當錨點，所以只能拿詞彙表比對。詞彙表從 `DOCS` 長出來，重建 `jobs.json` 之後會自動跟著更新。

詞彙表外的區名（例如「板橋有工作嗎？」）過濾不會觸發，退回純向量檢索——實測模型會正確回答「資料裡沒有」，所以這個邊界不需要額外處理。

分類全部是多值的——一個職缺可能同時是遠端和台南（`INT-01`），也可能同時收全職和兼職（`LEOSYS-AILAB-OT01`）。抓不到就留空，代表「資料沒寫」而不是「否」：中華創智那三個職缺的工作性質欄被填成了工作內容，那就誠實留空，不猜。

### 完整的清單由程式給，不是模型

檢索能保證完整，但**生成不能**。實測 Qwen2.5-3B 被要求列舉 20 筆時只列得出 16~17 筆，而且常常不照 `SYSTEM` 的指示附上代號——即使把代號放在每一筆 context 的第一行也一樣。調 prompt 和溫度都不可靠（同一組設定跑兩次結果不同）。

所以架構上分工：

- **`label` 以代號開頭**，UI 的來源列把每一筆 hit 都顯示出來——這是完整且精確的清單
- **模型的散文只當摘要看**

實測「有哪些台北的職缺」：模型散文列 15 筆，來源列 20 筆完整。

## 雲端的兩個配額陷阱

都是 FAQ 時代（5 筆）碰不到、換成職缺（140 塊）才炸出來的：

| 限制 | 症狀 | 處理 |
|------|------|------|
| `BatchEmbedContentsRequest` 一次最多 100 筆 | `400 INVALID_ARGUMENT` | 分批送 |
| 免費方案每分鐘 100 個 request，**batch 裡每段文字各算一個** | `429 RESOURCE_EXHAUSTED` | 從錯誤訊息讀 `retry in Ns` 自動重試 |

第二個代表建一次索引要 140 個 request、跨兩個配額視窗，實測耗時 **136 秒**。所以雲端這側加了快取 `data/vecs_cloud.npz`（421 KB，不進版控）——走快取只要 2.3 秒、零 API 呼叫。快取的 key 是所有 `text` 的 SHA256，重建 `jobs.json` 之後會自動失效重算。

地端不需要快取：e5 在本機跑，重算只是慢十秒，不燒任何配額。

## 後端 API

問答邏輯包成了 HTTP 服務，Streamlit 改成打它。好處是實質的：後端可以獨立
測試、可以被別的東西呼叫（`curl`、之後的 React、別的服務）、可以部署在另一台
機器；UI 掛掉也不會拖垮索引。

```bash
uvicorn api.main:app --reload      # 後端，聽 http://localhost:8000
streamlit run cloud_app.py         # 另開一個終端
```

互動式 API 文件在 <http://localhost:8000/docs>（FastAPI 從 `api/schemas.py` 自動生成）。

| 方法 | 路徑 | 說明 |
|---|---|---|
| GET | `/health` | 健康檢查，附帶「哪幾邊的索引已經算好」 |
| POST | `/chat` | 檢索 + 生成，一次回完 |
| POST | `/chat/stream` | 同上，但用 SSE 逐段吐字 |

兩個聊天端點吃同一組參數（`question`、`history`、`side`、`k`），也走同一份
`ChatBot`——差別只在回應怎麼送。

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json"      -d '{"question":"...","side":"onperm"}'
```

回應長這樣：

```json
{
  "reply": "資料中有以下無人機相關的職缺…",
  "sources": [{"code": "MAT-04", "label": "MAT-04 · 神耀科技 · 無人機硬體工程師 · 臺北市內湖區", "score": 0.906}],
  "usage": {"prompt": 2764, "output": 78},
  "history": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]
}
```

`code` 是獨立欄位而不是埋在 `reply` 的散文裡——[完整的清單由程式給，不是模型](#完整的清單由程式給不是模型)。

### 串流

`POST /chat/stream` 回的是 SSE，四種事件：

| 事件 | 內容 | 時機 |
|---|---|---|
| `sources` | 檢索結果 | **第一個送**。檢索比生成快得多，前端可以先把來源列出來 |
| `token` | 一段文字 | 模型每吐一段 |
| `done` | `usage` 與更新後的 `history` | 正常結束 |
| `error` | 人話的錯誤訊息 | 生成中途壞掉 |

實測（本機 Qwen2.5-3B）：來源 2.6 秒就到，首段同時，全文 4.4 秒 76 段。

**連線階段的錯誤仍然是正常的 HTTP 狀態碼。** 端點會先「預抽」第一段，抽得出來才
開始串流——代價是 `sources` 要等模型吐第一個字（實測 0.2 秒），換到的是 503／429
不會偽裝成 200。抽出來之後就沒辦法再改狀態碼了，所以那之後的失敗只能用 `error`
事件回報；**呼叫端一定要處理這個事件**，不然畫面會停在半截答案上。

### 串流踩到的兩個編碼坑

兩個都是「不會報錯、只會默默變成亂碼或 0」的那種：

- **`Content-Type` 一定要宣告 `charset=utf-8`。** SSE 是 `text/event-stream`，
  HTTP 對 `text/*` 的規定是沒宣告就退回 ISO-8859-1，`requests` 照做——中文會變成
  `'å¥½ç'`。伺服器端有宣告，客戶端仍再設一次當保險。
- **llama.cpp 在 `stream=True` 時預設不回 `usage`**，要明確加
  `stream_options: {"include_usage": true}` 才給。跟非串流那條路不一樣，
  忘了加就是 token 數永遠 0。

### 後端不記 history

`history` 由呼叫端帶進來、原樣帶回去，伺服器什麼都不存。

這不是偷懶。把對話歷史做成模組層單例的話，整個 process 共用一份——**兩個
使用者會看到彼此的對話，任一人清空就清掉所有人的**。綁在連線上可以解決，
但 REST 這層根本不需要連線的概念，交給呼叫端保管最單純：併發永遠不會互相
污染、伺服器可以隨時重啟、也能水平擴充。Streamlit 那側本來就把 history 放在
`st.session_state`，剛好對得上。

### 錯誤碼

`llm.explain()` 認得的都是「上游的可預期狀況」，翻成 429／503；500 留給
「我們自己壞了」——不然呼叫端沒辦法判斷該不該重試。

| 狀況 | 狀態碼 |
|---|---|
| 參數不合法（空問題、`side` 不存在、`k` 超出範圍） | 422 |
| Gemini 配額用完 | 429 |
| Gemini 過載、llama.cpp server 沒開或逾時、某一邊起不來 | 503 |
| 其他 | 500 |

### 索引什麼時候建

預設是「第一次用到那一邊才建」——只用雲端的人不該在啟動時等地端載 e5。
要預熱就設 `API_WARM=cloud` 或 `API_WARM=cloud,onperm`。

預熱會**真的把索引算出來**，不只是把物件建起來——後者幾乎不花時間，講出來
沒有意義。實測地端第一個請求：沒預熱 47.6 秒，預熱後 2.6 秒。

`/health` 的 `loaded` 回報的也是「索引算好了」而不是「物件建好了」，
所以看得出預熱有沒有生效。

## 測試與 lint

```bash
python -m pytest          # 179 個測試，約 2 秒
python -m ruff check .    # lint
pre-commit install        # 裝一次，之後每次 commit 自動跑上面兩項
```

測試刻意「不碰模型也不碰網路」，所以整批跑完是秒級的：

| 檔案 | 測什麼 | 怎麼避開外部依賴 |
|------|--------|-----------------|
| `test_facets.py` | 正規化與查詢條件抽取 | 純函式，本來就沒有依賴 |
| `test_build_jobs.py` | PDF 文字清理、切塊、欄位解析 | 假的 PDF 物件 |
| `test_retriever.py` | 檢索邏輯：去重、門檻、過濾放寬 | 分數由字典指定的假檢索器 |
| `test_chat_bot.py` | 編排：prompt 組裝、history 回滾、短路、串流 | 假的 `BaseLLM` 子類 |
| `test_llm.py` | 各家的線路格式轉換、`usage`、錯誤翻譯、串流 | 假 client／換掉 `requests.post` |
| `test_settings.py` | 預設值、環境變數覆寫、型別轉換 | `_env_file=None` 不讀本機 `.env` |
| `test_api.py` | 端點：驗證、`Doc`→`Source`、狀態碼、無狀態、SSE 事件 | `dependency_overrides` 換掉整個 bot |
| `test_api_client.py` | 客戶端：請求形狀、錯誤訊息可不可讀 | 換掉 `requests.request` |
| `test_knowledge.py` | `Doc` 預設值、`import` 不讀檔 | — |
| `test_corpus.py` | 對真實 `jobs.json` 的筆數回歸 | 過濾是 facet 決定的，不需要算向量 |

`test_chat_bot.py` 與 `test_llm.py` 的分工就是 Stage 2 的成果：編排邏輯只有一份、
測一次；線路格式的差異各自關在自己的 provider 測試裡。以前這些要寫兩遍。

`test_retriever.py` 是 Stage 0 的直接成果——在那之前 `DOCS` 和索引都是模組層全域，
沒辦法塞一批自己的資料進去，這些測試寫不出來。

`test_corpus.py` 把 [metadata 過濾](#metadata-過濾) 那張表釘成測試：問句撈回幾個職缺
完全由 facet 決定、跟相似度無關，所以不必載 embedding 模型也驗得出來。
**重建 `jobs.json` 之後這些數字會變，要跟 README 一起更新。**

### 兩個工具設定上的決定

- **不跑 `ruff format`**。這份程式碼的行內註解是手動對齊的（欄位說明、指向特定參數的箭頭），
  自動排版會把對齊全部打散，換來的只有風格統一。所以只 lint，不排版。
- **`UP031` 關掉**。ruff 想把 13 處 `%` 格式化改成 f-string，但參數多或字串含大量中文時
  `%` 讀起來比較清楚。全部改寫是純風格 churn。
- `.pre-commit-config.yaml` 是全專案唯一用英文註解的檔案：pre-commit 用系統 locale
  （這台機器是 cp950）而不是 UTF-8 讀它，寫中文會讓它在解析 YAML 之前就 UnicodeDecodeError。

## 說明

`.env` 已列入 `.gitignore`，金鑰不會進版控。模型檔（`*.gguf`、`models/`）也不進版控。
