import { useCallback, useEffect, useRef, useState } from "react";

import { API_BASE, health as fetchHealth, type Health, type Side } from "./api";
import { Composer } from "./components/Composer";
import { Documents } from "./components/Documents";
import { Message } from "./components/Message";
import { useChat } from "./hooks/useChat";
import { useDocuments } from "./hooks/useDocuments";

const TITLES: Record<Side, string> = { cloud: "雲端（Gemini）", onperm: "地端（llama.cpp）" };

// 空狀態的例句。挑的是三種不同的問法：關鍵字、公司名、篩選型——
// 篩選型那一句會走 metadata 過濾，回的是完整清單而不是前 5 名。
const EXAMPLES = ["有哪些台北的職缺？", "耐能智慧在徵什麼人？", "有哪些兼職的職缺？"];

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

  // 新內容進來就捲到底。桌機是對話欄自己捲；≤833px 版面攤平成整頁捲動，
  // 那時 .viewport 不是捲動容器，scrollTo 會是空操作——所以改捲視窗。
  useEffect(() => {
    const el = viewport.current;
    if (!el) return;
    if (el.scrollHeight > el.clientHeight) el.scrollTo({ top: el.scrollHeight });
    else window.scrollTo({ top: document.body.scrollHeight });
  }, [chat.history, chat.pending]);

  const loaded = health?.loaded.includes(side) ?? false;
  const chunks = health?.chunks[side];
  const indexLine = !health
    ? "索引未知"
    : chunks == null
      ? "索引未載入"
      : `索引 ${chunks.toLocaleString()} 塊・${health.documents} 份文件`;

  return (
    <div className="app">
      {/* 全域導覽列：純黑 44px。整個頁面只有這裡出現純黑。 */}
      <nav className="global-nav">
        <span className="wordmark">職缺媒合</span>
        <span className="nav-right">
          <span className={`dot ${loaded ? "ok" : "off"}`} />
          <span>{health ? TITLES[side] : "連不上後端"}</span>
        </span>
      </nav>

      {/* 次導覽：羊皮紙毛玻璃。左邊是這個畫面叫什麼，右邊是工具。 */}
      <div className="sub-nav">
        <h1 className="t-tagline">媒合會職缺查詢</h1>
        <div className="sub-nav-right">
          <span className="meta t-caption">{indexLine}</span>
          <button
            className="btn-utility"
            onClick={chat.reset}
            disabled={chat.busy || !chat.history.length}
          >
            清空對話
          </button>
        </div>
      </div>

      <div className="workspace">
        <main className="thread">
          <div className="viewport" ref={viewport}>
            {!chat.history.length && !chat.pending ? (
              // 空狀態就是一塊 hero 磚：大標、一行 lead、幾顆例句晶片。
              <div className="hero">
                <h2 className="t-hero">問一句，職缺自己浮出來。</h2>
                <p className="t-lead">直接用中文問地點、公司或工作性質，不必想關鍵字。</p>
                <div className="chip-row">
                  {EXAMPLES.map((q) => (
                    <button key={q} className="chip" onClick={() => chat.send(q)}>
                      {q}
                    </button>
                  ))}
                </div>
                <p className="hero-note t-caption">
                  篩選型的問句（地點、工作性質、學歷）會走 metadata 過濾，
                  回傳的是<strong>完整清單</strong>而不是前 5 名。
                </p>
                {!health && (
                  <p className="hero-note t-caption">
                    後端還沒起來：<code>uvicorn api.main:app --reload</code>
                  </p>
                )}
              </div>
            ) : (
              <div className="thread-inner">
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
                        content={chat.pending.text}
                        sources={chat.pending.sources}
                        streaming={chat.busy}
                      />
                    )}
                    {chat.pending.failed && (
                      <div className="inset">
                        <span className="inset-label t-caption-strong">連線中斷</span>
                        <span className="inset-body t-caption">{chat.pending.failed}</span>
                        <button className="link t-caption" onClick={chat.dismiss}>
                          關閉
                        </button>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>

          <Composer busy={chat.busy} onSend={chat.send} />
        </main>

        <aside className="sidebar">
          <section className="card">
            <h2>推論在哪裡跑</h2>
            <div className="toggle">
              {(["cloud", "onperm"] as Side[]).map((s) => (
                <button
                  key={s}
                  className="chip"
                  aria-pressed={side === s}
                  disabled={chat.busy}
                  onClick={() => setSide(s)}
                >
                  {s === "cloud" ? "雲端" : "地端"}
                </button>
              ))}
            </div>
            <p className="note t-caption">
              對話格式兩邊一樣，所以中途換邊也接得下去。
              {health && ` 已載入：${health.loaded.join("、") || "無"}`}
            </p>
          </section>

          {/* 暗磚：跟上下兩張白卡交替，色彩變化本身就是分隔線。 */}
          <section className="card tile">
            <h2>Token 用量</h2>
            <dl className="stats">
              <div>
                <dt>輸入</dt>
                <dd>{chat.usage.prompt.toLocaleString()}</dd>
              </div>
              <div>
                <dt>輸出</dt>
                <dd>{chat.usage.output.toLocaleString()}</dd>
              </div>
            </dl>
            <p className="note t-caption">
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

          <footer className="footer t-fine">
            <p>後端 {API_BASE}</p>
            <p>對話留在瀏覽器裡——後端是無狀態的，每次把 history 帶上去。</p>
          </footer>
        </aside>
      </div>
    </div>
  );
}
