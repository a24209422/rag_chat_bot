import { useRef, useState } from "react";

interface Props {
  busy: boolean;
  onSend: (question: string) => void;
}

/** 輸入列 = 文件裡的 floating-sticky-bar（羊皮紙 80% + 毛玻璃）裡面放一個
 *  膠囊搜尋框。送出鈕就坐在膠囊右端——pill 在這套語彙裡是「行動」的專屬訊號。 */
export function Composer({ busy, onSend }: Props) {
  const [text, setText] = useState("");
  const box = useRef<HTMLTextAreaElement>(null);

  // 高度跟著內容長。送出後也要叫一次，不然膠囊會停在多行的高度上空著。
  function fit() {
    const el = box.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }

  function submit() {
    const question = text.trim();
    if (!question || busy) return;
    onSend(question);
    setText("");
    // 送出後直接還原成 auto（textarea 的 auto = rows 那一行的高度），不必等
    // React 把 value 清掉再量——那一幀之前膠囊會停在多行的高度上空著。
    if (box.current) box.current.style.height = "auto";
  }

  return (
    <div className="composer-bar">
      <div className="composer">
        {/* placeholder 保持短句：手機版膠囊只有 300 多 px，長句會折成兩行撐破形狀。 */}
        <textarea
          ref={box}
          rows={1}
          value={text}
          placeholder="想找什麼樣的職缺？"
          disabled={busy}
          onChange={(e) => {
            setText(e.target.value);
            fit();
          }}
          onKeyDown={(e) => {
            // isComposing：用注音／拼音打字時，Enter 是「確認候選字」而不是
            // 「送出」。不擋的話中文使用者每選一次字就送出一則半截訊息。
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <button className="btn-primary" onClick={submit} disabled={busy || !text.trim()}>
          {busy ? "思考中…" : "送出"}
        </button>
      </div>
    </div>
  );
}
