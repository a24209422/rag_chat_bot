# 前端（React + Vite + TypeScript）

後端的 Streamlit 介面還在（`streamlit run cloud_app.py`），這裡是另一個客戶端。
兩者打同一組 API，所以沒有重複的問答邏輯——差別只在畫面。

## 跑起來

```bash
npm install
npm run dev          # http://localhost:5173
```

需要後端在 `http://localhost:8000`：

```bash
uvicorn api.main:app --reload
```

要改後端位址就複製 `.env.example` 成 `.env` 改 `VITE_API_URL`。

CORS 不用管——後端放行 `localhost`／`127.0.0.1` 的**任意埠**，所以 5173 被佔走
跳 5174、或 `npm run preview`（4173）都打得進去。只有從別台機器連進來
（`npm run dev -- --host`，用手機或另一台電腦開）才要把那個來源加進後端的
`CORS_ORIGINS`；沒加的症狀是預檢被擋，後端日誌會寫出是哪個來源。

## 指令

| | |
|---|---|
| `npm run dev` | 開發伺服器 |
| `npm run build` | `tsc -b` + 正式建置到 `dist/` |
| `npm run lint` | oxlint |

## 檔案

```
src/
├── api.ts                  後端的唯一入口：型別、REST、SSE
├── App.tsx                 版面
├── index.css               設計系統（色票／字級／圓角／間距都是 CSS 變數）
├── components/
│   ├── Message.tsx         一則訊息 + 來源列
│   ├── Composer.tsx        輸入框
│   └── Documents.tsx       文件上傳與管理
└── hooks/
    ├── useChat.ts          對話狀態與串流
    └── useDocuments.ts     文件清單
```

## 設計系統

介面照 [`DESIGN-apple.md`](DESIGN-apple.md) 實作（那份是設計規格的原文，
一起放進版控免得規則跟著某台機器走），規則落在 `src/index.css` 最上面的變數區：

| | |
|---|---|
| 強調色 | 只有 Action Blue `#0066cc` 一個。所有「可點」的東西都是它，沒有第二色 |
| 底色 | 羊皮紙 `#f5f5f7` + 白卡（18px 圓角、1px 髮絲線） |
| 內文 | 17px / 1.47 / -0.374px——不是 16px |
| 陰影 | 沒有。原始系統只有一個陰影而且保留給產品照，這裡沒有產品照 |
| 圓角 | 8px 工具鈕、11px 珍珠膠囊、18px 卡片、pill 只給行動鈕。不混用 |
| 層次 | 靠換底色（亮磚↔暗磚）與 `backdrop-filter`，不靠邊框陰影 |

兩個地方是照文件的邏輯往外推的，文件本身沒有寫：

- **錯誤與警示是一塊翻面的磚**（亮模式翻近黑、暗模式翻羊皮紙）。原始系統
  沒有第二個顏色可以借，而它給的答案是「要強調就換底色」。
- **暗色模式**用文件裡的近黑磚色（`#272729` / `#2a2a2c` / `#252527`），
  深色底上的連結換成 Sky Link Blue `#2997ff`。原文件只記錄了亮色版。

字體：macOS／iOS 走系統的 SF Pro，其他平台退到 Inter（`index.html` 從
Google Fonts 拉，文件指定的替代品）。**拉不到不會壞**，只是少一點那個味道；
要完全離線就把 `index.html` 裡那三行 `<link>` 刪掉。中文一律由系統字型接手。

## 幾個要知道的地方

**SSE 用 `fetch` 而不是 `EventSource`**。`EventSource` 只能 GET，而我們要送
history。所以是 `fetch` + `ReadableStream`，自己切 `\n\n` 分事件。

`TextDecoder` 一定要用 `{ stream: true }`——一個中文字是三個位元組，可能被切在
兩個 chunk 中間，不加這個參數就會吐出半個字。

**history 由前端保管**。後端無狀態：每次把 history 送進去、拿回更新後的版本。
併發使用者不會互相污染。失敗的那一輪不進 history（後端也會 rollback），
但半截答案留在畫面上，使用者才知道發生了什麼事。

**Enter 送出要擋 `isComposing`**。用注音／拼音打字時 Enter 是「確認候選字」，
不擋的話中文使用者每選一次字就送出一則半截訊息。

**沒有用 Tailwind**。這個規模用不到一整套 utility class，手寫 CSS 變數少一層
build 外掛。要換 Tailwind 的話 `index.css` 是唯一要改的地方。
