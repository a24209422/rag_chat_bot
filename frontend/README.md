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

要改後端位址就複製 `.env.example` 成 `.env` 改 `VITE_API_URL`，
並把前端的來源加進後端的 `CORS_ORIGINS`。

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
├── index.css               全域樣式（CSS 變數，亮/暗跟系統走）
├── components/
│   ├── Message.tsx         一則訊息 + 來源列
│   ├── Composer.tsx        輸入框
│   └── Documents.tsx       文件上傳與管理
└── hooks/
    ├── useChat.ts          對話狀態與串流
    └── useDocuments.ts     文件清單
```

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
