import { useCallback, useState } from "react";

import { ApiError, chatStream, type Message, type Side, type Source, type Usage } from "../api";

/** 串流中（或剛失敗）的那一輪。還沒進 history。 */
export interface Pending {
  question: string;
  text: string;
  sources: Source[];
  failed?: string;
}

export function useChat(side: Side) {
  const [history, setHistory] = useState<Message[]>([]);
  // 與 history 等長；user 那格是 null。後端可能就地截短 history（地端有上限），
  // 所以每輪結束後取相同長度的尾段對齊——不然來源會標到別人的回答底下。
  const [sources, setSources] = useState<(Source[] | null)[]>([]);
  const [usage, setUsage] = useState<Usage>({ prompt: 0, output: 0 });
  const [pending, setPending] = useState<Pending | null>(null);
  const [busy, setBusy] = useState(false);

  const send = useCallback(
    async (question: string) => {
      if (busy || !question.trim()) return;
      setBusy(true);
      setPending({ question, text: "", sources: [] });

      // onSources 一定在 onDone 之前，所以區域變數就夠——不必為它開一個 state。
      let turnSources: Source[] = [];

      try {
        await chatStream(
          { side, question, history },
          {
            onSources: (got) => {
              turnSources = got;
              setPending((p) => (p ? { ...p, sources: got } : p));
            },
            onToken: (text) =>
              setPending((p) => (p ? { ...p, text: p.text + text } : p)),
            onDone: ({ usage: used, history: updated }) => {
              setHistory(updated);
              setSources((prev) => {
                const next = [...prev, null, turnSources];
                return updated.length ? next.slice(-updated.length) : [];
              });
              setUsage((u) => ({
                prompt: u.prompt + used.prompt,
                output: u.output + used.output,
              }));
              setPending(null);
            },
          },
        );
      } catch (e) {
        // 失敗的那一輪「不進」history——後端也把它收回去了（Turn 會 pop），
        // 兩邊要一致，不然下一輪送出去的上下文會多一則不存在的問答。
        // 但半截答案留在畫面上，使用者才知道發生了什麼事。
        setPending((p) => ({
          question,
          text: p?.text ?? "",
          sources: p?.sources ?? [],
          failed: e instanceof ApiError ? e.message : String(e),
        }));
      } finally {
        setBusy(false);
      }
    },
    [busy, history, side],
  );

  const reset = useCallback(() => {
    setHistory([]);
    setSources([]);
    setPending(null);
  }, []);

  return { history, sources, usage, pending, busy, send, reset, dismiss: () => setPending(null) };
}
