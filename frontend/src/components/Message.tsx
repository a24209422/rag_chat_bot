import type { Source } from "../api";

interface Props {
  role: "user" | "assistant";
  content: string;
  sources?: Source[] | null;
  contradiction?: boolean | null;
  streaming?: boolean;
}

/** label 本身以代號開頭（"代號 · 公司 · 職缺 · …"），代號又是獨立欄位。
 *  兩邊都印會重複，所以把開頭那一段剝掉，代號單獨排一欄當規格表的列首。 */
function trimCode(label: string, code: string) {
  const rest = label.startsWith(code) ? label.slice(code.length) : label;
  return rest.replace(/^\s*[·・|-]\s*/, "");
}

export function Message({ role, content, sources, contradiction, streaming }: Props) {
  const hasSources = !!sources && sources.length > 0;

  return (
    <div className={`msg ${role}`}>
      <div className={`bubble${hasSources ? " has-sources" : ""}`}>
        {/* 游標要跟在字的後面，不是跟在來源表後面——串流時來源比字先到。 */}
        <div className={`text${streaming ? " caret" : ""}`}>{content}</div>

        {/* 來源列：這一列才是完整且精確的清單。模型的散文只當摘要看——實測
            3B 模型被要求列 20 筆時只列得出 16~17 筆，而且常常不附代號。 */}
        {hasSources && (
          <div className="sources">
            <div className="sources-head t-caption-strong">檢索到 {sources.length} 筆</div>
            {sources.map((s) => (
              <div className="source t-caption" key={s.code}>
                <span className="code">{s.code}</span>
                <span className="label">{trimCode(s.label, s.code)}</span>
                <span className="score">{s.score.toFixed(3)}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 模型說沒有、檢索卻有。後端重抽一次之後還是這樣才會走到這裡（約 1%）。
          不是錯誤，是「別信那句話，看上面那張表」——所以不用錯誤色，用翻面的
          磚：這套語彙裡「換底色」就是強調本身。 */}
      {contradiction && (
        <div className="inset">
          <span className="inset-label t-caption-strong">以來源為準</span>
          <span className="inset-body t-caption">
            模型說資料裡沒有，但檢索到了 {sources?.length ?? 0} 筆。
          </span>
        </div>
      )}
    </div>
  );
}
