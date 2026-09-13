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
    <section className="card docs">
      <h2>知識庫文件</h2>

      {/* 錯誤是一塊翻面的磚，不是紅底紅字——這套語彙沒有第二個顏色。 */}
      {error && (
        <div className="inset" style={{ marginBottom: "var(--s-sm)" }}>
          <span className="inset-body t-caption">{error}</span>
          <button className="link t-caption" onClick={onDismissError}>
            關閉
          </button>
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
        <span className="btn-pearl">{busy ? "處理中…" : "上傳職缺 PDF"}</span>
      </label>

      {docs.length > 0 && (
        <ul className="doc-list">
          {docs.map((d) => (
            <li className="doc t-caption" key={d.doc_id}>
              <span className="doc-name" title={d.filename}>
                {d.filename}
              </span>
              <span className="doc-meta">{d.jobs} 個職缺</span>
              <button className="link" onClick={() => onRemove(d.doc_id)} disabled={busy}>
                刪除
              </button>
            </li>
          ))}
        </ul>
      )}

      <p className="note t-caption">
        只吃媒合會職缺表格式的 PDF。同名重傳視為更新；基礎語料不在這裡，也刪不掉。
      </p>
    </section>
  );
}
