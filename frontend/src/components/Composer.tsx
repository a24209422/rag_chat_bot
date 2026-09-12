import { useState } from "react";

interface Props {
  busy: boolean;
  onSend: (question: string) => void;
}

export function Composer({ busy, onSend }: Props) {
  const [text, setText] = useState("");

  function submit() {
    const question = text.trim();
    if (!question || busy) return;
    onSend(question);
    setText("");
  }

  return (
    <div className="composer">
      <textarea
        rows={1}
        value={text}
        placeholder="說點什麼…（Enter 送出，Shift+Enter 換行）"
        disabled={busy}
        onChange={(e) => {
          setText(e.target.value);
          e.target.style.height = "auto";
          e.target.style.height = `${e.target.scrollHeight}px`;
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
      <button onClick={submit} disabled={busy || !text.trim()}>
        {busy ? "思考中…" : "送出"}
      </button>
    </div>
  );
}
