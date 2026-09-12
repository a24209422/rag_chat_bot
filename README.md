# RAG Chat-bot 實作

一組最小可跑的 RAG（Retrieval-Augmented Generation）聊天機器人練習，同一套流程各做了**雲端版**與**地端版**兩種實作，方便對照兩者的差異。

知識庫是 11 家公司的媒合會職缺表（PDF），共 30 個職缺。

## 架構

RAG 拆成兩段：先用 embedding 從知識庫撈出最相關的段落，再把段落和問題一起丟給 LLM 回答。

雲端與地端並排成兩個資料夾。兩邊的檢索器繼承同一個 base class，所以「同名同形狀」是
繼承保證的，不是靠約定；換邊只要換 `providers.chat_for()` 的參數：

```
├── providers.py            retriever_for() / chat_for()：依名稱取得某一邊的實作
├── cloud_app.py            Streamlit 介面（雲端版）
├── onperm_app.py           Streamlit 介面（地端版）
├── data/
│   └── jobs.json           知識庫（由 tools.build_jobs 產生）
├── shared/
│   ├── knowledge.py        Doc 結構 + load_docs() + SYSTEM，兩邊共用
│   ├── retriever.py        BaseRetriever：共用的檢索邏輯，子類只補 embed()
│   └── facets.py           把自由文字的 metadata 正規化成可精確比對的分類
├── cloud/
│   ├── client.py           Gemini client 的唯一建構點（讀金鑰只有這一處）
│   ├── rag.py              CloudRetriever：gemini-embedding-001，維度截短成 768
│   └── chat_bot.py         CloudChatBot：gemini-flash-latest，多輪記憶與 token 計量
├── onperm/
│   ├── rag.py              OnpremRetriever：intfloat/multilingual-e5-small
│   └── chat_bot.py         OnpremChatBot：走 llama.cpp server 的 OpenAI 相容 API
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

`ask()` 雲端回傳 `(reply, hits, usage)`、地端只有 `(reply, hits)`（本機推論不計費）；history 格式一邊是 Gemini 的 `parts`/`model`、一邊是 OpenAI 的 `content`/`assistant`；地端會截短 history 防爆 context，雲端不用。

所以 `CloudChatBot` 和 `OnpremChatBot` 刻意沒有共同的 base class——硬合併只會換來一堆 `if side == ...`。反過來說 `retrieve()` 兩邊本來就逐字相同，那個才該合併，現在收在 `shared/retriever.py`，雲地差異縮到只剩 `embed()`、`min_score`、要不要快取索引三樣。

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
python -m cloud.rag           # 只測檢索
python -m cloud.chat_bot      # 互動對話（CLI）
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
python -m onperm.rag          # 只測檢索（不需要 server）
python -m onperm.chat_bot     # 互動對話（CLI，需要 server）
streamlit run onperm_app.py   # 網頁介面（需要 server）
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

## 說明

`.env` 已列入 `.gitignore`，金鑰不會進版控。模型檔（`*.gguf`、`models/`）也不進版控。
