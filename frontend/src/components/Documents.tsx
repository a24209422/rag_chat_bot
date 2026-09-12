import { useRef } from "react";

import type { DocumentOut } from "../api";

interface Props {
  docs: DocumentOut[];
  busy: boolean;
  error: string | null;
  onUpload: (file: File) => void;
  onRemove: (docId: string) => void;
  onDismissError: () => void;
}

export function Documents({ docs, busy, error, onUpload, onRemove, onDismissError }: Props) {
  const input = useRef<HTMLInputElement>(null);

  return (
    <section className="panel card">
      <h2>知識庫文件</h2>

      {error && (
        <div className="error" style={{ marginBottom: 10 }}>
          <span>{error}</span>
          <button onClick={onDismissError}>關閉</button>
        </div>
      )}

      <label className="upload">
        <input
          ref={input}
          type="file"
          accept=".pdf"
          disabled={busy}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onUpload(file);
            // 清掉，不然同一個檔案選第二次不會觸發 change
            if (input.current) input.current.value = "";
          }}
        />
        {busy ? "處理中…" : "上傳職缺 PDF"}
      </label>

      {docs.length > 0 && (
        <ul className="doc-list" style={{ marginTop: 10 }}>
          {docs.map((d) => (
            <li className="doc" key={d.doc_id}>
              <span className="doc-name" title={d.filename}>
                {d.filename}
              </span>
              <span className="doc-meta">{d.jobs} 個職缺</span>
              <button onClick={() => onRemove(d.doc_id)} disabled={busy}>
                刪除
              </button>
            </li>
          ))}
        </ul>
      )}

      <p className="note">
        只吃媒合會職缺表格式的 PDF。同名重傳視為更新；基礎語料不在這裡，也刪不掉。
      </p>
    </section>
  );
}
