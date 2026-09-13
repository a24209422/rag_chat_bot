import type { Source } from "../api";

interface Props {
  role: "user" | "assistant";
  content: string;
  sources?: Source[] | null;
  contradiction?: boolean | null;
  streaming?: boolean;
}

export function Message({ role, content, sources, contradiction, streaming }: Props) {
  return (
    <div className={`msg ${role}`}>
      <div className={`bubble${streaming ? " caret" : ""}`}>{content}</div>
      {/* 模型說沒有、檢索卻有。後端重抽一次之後還是這樣才會走到這裡。
          該相信的是來源列，所以把使用者的視線導過去。 */}
      {contradiction && (
        <div className="warn">
          模型說資料裡沒有，但檢索到了 {sources?.length ?? 0} 筆——以下面的來源為準。
        </div>
      )}
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
