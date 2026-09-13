import { useCallback, useEffect, useRef, useState } from "react";

import { health as fetchHealth, type Health, type Side } from "./api";
import { Composer } from "./components/Composer";
import { Documents } from "./components/Documents";
import { Message } from "./components/Message";
import { useChat } from "./hooks/useChat";
import { useDocuments } from "./hooks/useDocuments";

const TITLES: Record<Side, string> = { cloud: "雲端（Gemini）", onperm: "地端（llama.cpp）" };

export default function App() {
  const [side, setSide] = useState<Side>("cloud");
  const [health, setHealth] = useState<Health | null>(null);
  const chat = useChat(side);
  const documents = useDocuments();
  const viewport = useRef<HTMLDivElement>(null);

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await fetchHealth());
    } catch {
      setHealth(null); // 後端沒開。錯誤訊息由對話那條路顯示，這裡只要變灰
    }
  }, []);

  useEffect(() => {
  // 這是「跟外部系統同步」，不是同步 setState：setState 發生在 await 之後。
  // React 官方文件把「取遠端資料」列為 effect 的正當用途。
  // oxlint-disable-next-line react/set-state-in-effect
    void refreshHealth();
  }, [refreshHealth, documents.docs.length, chat.history.length]);

  // 新內容進來就捲到底
  useEffect(() => {
    viewport.current?.scrollTo({ top: viewport.current.scrollHeight });
  }, [chat.history, chat.pending]);

  const loaded = health?.loaded.includes(side) ?? false;

  return (
    <div className="app">
      <main className="panel chat">
        <header className="chat-header">
          <h1>媒合會職缺查詢</h1>
          <span className={`dot ${loaded ? "ok" : "off"}`} />
          <span className="doc-meta">
            {!health
              ? "連不上後端"
              : health.chunks[side] == null
                ? `${TITLES[side]}・索引未載入`
                : `${TITLES[side]}・索引 ${health.chunks[side]} 塊`}
          </span>
          <button onClick={chat.reset} disabled={chat.busy || !chat.history.length}>
            清空對話
          </button>
        </header>

        <div className="viewport" ref={viewport}>
          {!chat.history.length && !chat.pending && (
            <div className="empty">
              <p>問問看「有哪些台北的職缺？」或「耐能智慧在徵什麼人？」</p>
              <p>
                篩選型的問句（地點、工作性質、學歷）會走 metadata 過濾，
                回傳的是<strong>完整清單</strong>而不是前 5 名。
              </p>
              <p>
                需要後端：<code>uvicorn api.main:app --reload</code>
              </p>
            </div>
          )}

          {chat.history.map((m, i) => (
            <Message
              key={i}
              role={m.role}
              content={m.content}
              sources={chat.sources[i]}
              contradiction={chat.flags[i]}
            />
          ))}

          {chat.pending && (
            <>
              <Message role="user" content={chat.pending.question} />
              {/* 連一個字都還沒吐就失敗的話，不要留一個空泡泡在那裡——
                  錯誤訊息自己會說明發生什麼事。 */}
              {(chat.pending.text || !chat.pending.failed) && (
                <Message
                  role="assistant"
                  content={chat.pending.text || "…"}
                  sources={chat.pending.sources}
                  streaming={chat.busy}
                />
              )}
              {chat.pending.failed && (
                <div className="error">
                  <span>{chat.pending.failed}</span>
                  <button onClick={chat.dismiss}>關閉</button>
                </div>
              )}
            </>
          )}
        </div>

        <Composer busy={chat.busy} onSend={chat.send} />
      </main>

      <aside className="sidebar">
        <section className="panel card">
          <h2>推論在哪裡跑</h2>
          <div className="toggle">
            {(["cloud", "onperm"] as Side[]).map((s) => (
              <button
                key={s}
                aria-pressed={side === s}
                disabled={chat.busy}
                onClick={() => setSide(s)}
              >
                {s === "cloud" ? "雲端" : "地端"}
              </button>
            ))}
          </div>
          <p className="note">
            對話格式兩邊一樣，所以中途換邊也接得下去。
            {health && ` 已載入：${health.loaded.join("、") || "無"}`}
          </p>
        </section>

        <section className="panel card">
          <h2>Token 用量</h2>
          <dl className="stats">
            <dt>輸入</dt>
            <dd>{chat.usage.prompt.toLocaleString()}</dd>
            <dt>輸出</dt>
            <dd>{chat.usage.output.toLocaleString()}</dd>
          </dl>
          <p className="note">
            雲端的輸出含模型的思考 token——看不到但會計費。
            地端不計費，但看得出離 context 上限還有多遠。
          </p>
        </section>

        <Documents
          docs={documents.docs}
          busy={documents.busy}
          error={documents.error}
          onUpload={documents.upload}
          onRemove={documents.remove}
          onDismissError={documents.dismissError}
        />
      </aside>
    </div>
  );
}
