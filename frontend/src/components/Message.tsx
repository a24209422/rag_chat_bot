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
  // 第一個字還沒到的那段空窗。它不只是網路延遲：後端會先預抽第一段才開始回應，
  // 偵測到矛盾時整段押著重問（約 11%，等於兩次生成的時間），地端第一次請求還要
  // 先把索引建出來（實測 47 秒）。空泡泡配一個閃動的游標看起來像壞掉了。
  const waiting = !!streaming && !content;

  return (
    <div className={`msg ${role}`}>
      <div className={`bubble${hasSources ? " has-sources" : ""}`}>
        {/* 游標要跟在字的後面，不是跟在來源表後面——串流時來源比字先到。 */}
        {waiting ? (
          <div className="thinking t-caption">思考中…</div>
        ) : (
          <div className={`text${streaming ? " caret" : ""}`}>{content}</div>
        )}

        {/* 來源列：這一列才是完整且精確的清單。模型的散文只當摘要看——實測
            3B 模型被要求列 20 筆時只列得出 16~17 筆，而且常常不附代號。

            預設收合，留「檢索到 N 筆」那一行。sources 事件比 token 先到（檢索
            快得多），照實畫的話畫面會先長出一張 20 列的表，答案再從表的上面
            慢慢擠出來——先看到、佔最多面積的東西反而不是使用者在等的那個。
            筆數那一行是「檢索有撈到東西」的即時回饋，一行不搶戲。

            用原生 <details>／<summary>：鍵盤操作、展開狀態、瀏覽器的頁內搜尋
            都免費得到，不必自己管 state。生成途中也點得開——來源那時候早就
            到了，沒有理由鎖著。

            ⚠ 生成途中展開的話，吐完字的那一刻會收回去：那一則訊息從 pending
              換成 history，是不同的節點，<details> 的 open 跟著舊節點走了。
              要保住的話得把展開狀態提到 useChat 裡按輪次記，不值得。 */}
        {hasSources && (
          <details className="sources">
            <summary className="sources-head t-caption-strong">
              檢索到 {sources.length} 筆
            </summary>
            {sources.map((s) => (
              <div className="source t-caption" key={s.code}>
                <span className="code">{s.code}</span>
                <span className="label">{trimCode(s.label, s.code)}</span>
                <span className="score">{s.score.toFixed(3)}</span>
              </div>
            ))}
          </details>
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
