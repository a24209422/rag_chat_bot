import type { Source } from "../api";

interface Props {
  role: "user" | "assistant";
  content: string;
  sources?: Source[] | null;
  streaming?: boolean;
}

export function Message({ role, content, sources, streaming }: Props) {
  return (
    <div className={`msg ${role}`}>
      <div className={`bubble${streaming ? " caret" : ""}`}>{content}</div>
      {sources && sources.length > 0 && (
        <div className="sources">
          {/* 這一列才是完整且精確的清單。模型的散文只當摘要看——實測 3B 模型
              被要求列 20 筆時只列得出 16~17 筆，而且常常不附代號。 */}
          {sources.map((s) => (
            <div className="source" key={s.code}>
              <span className="score">{s.score.toFixed(3)}</span>
              <span>{s.label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
