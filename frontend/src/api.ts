// api.ts（打後端的唯一入口）
//
// 後端是無狀態的：history 由這裡保管、每次帶進去、拿回更新後的版本。
// 併發使用者永遠不會看到彼此的對話。

const BASE = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

/** 頁尾要顯示現在打的是哪一台——設錯 VITE_API_URL 時這一行最快看出來。 */
export const API_BASE = BASE;

export type Side = "cloud" | "onperm";

export interface Message {
  role: "user" | "assistant";
  content: string;
}

export interface Source {
  code: string; // 職缺代號。應徵表單要填這個，所以它是獨立欄位，
  label: string; //   不是埋在 reply 的散文裡——模型記不住 20 筆的代號
  score: number;
}

export interface Usage {
  prompt: number;
  output: number; // 雲端這個數字含思考 token——看不到但會計費
}

export interface DocumentOut {
  doc_id: string;
  filename: string;
  content_type: string;
  size: number;
  version: string;
  jobs: number;
  created_at: string;
}

export interface Health {
  status: "ok";
  sides: Side[];
  loaded: Side[];
  chunks: Partial<Record<Side, number>>;
  documents: number;
}

export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** FastAPI 的錯誤是 {"detail": ...}，但 422 的 detail 是一個清單（每個欄位一則）。 */
async function detailOf(res: Response): Promise<string> {
  try {
    const body = await res.json();
    const detail = body?.detail;
    if (Array.isArray(detail)) return detail.map((d) => d?.msg ?? String(d)).join("；");
    if (detail) return String(detail);
  } catch {
    // 不是 JSON——例如反向代理回了一頁 HTML
  }
  return `後端回了 ${res.status}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(BASE + path, init);
  } catch {
    // fetch 只有在網路層失敗時才 reject，狀態碼再糟都算成功
    throw new ApiError(`連不上後端 ${BASE}——先跑 uvicorn api.main:app --reload`);
  }
  if (!res.ok) throw new ApiError(await detailOf(res), res.status);
  return res.json() as Promise<T>;
}

export const health = () => request<Health>("/health");
export const listDocuments = () => request<DocumentOut[]>("/documents");
export const deleteDocument = (docId: string) =>
  request<unknown>(`/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });

export function uploadDocument(file: File) {
  const body = new FormData();
  body.append("file", file);
  return request<{ document: DocumentOut }>("/documents", { method: "POST", body });
}

interface StreamHandlers {
  onSources?: (sources: Source[]) => void;
  onToken?: (text: string) => void;
  // contradiction：模型說「資料裡沒有」但 sources 不是空的。後端已經為此
  // 重抽過一次，這個旗標代表「重抽完還是矛盾」——該提醒使用者看來源列。
  onDone?: (result: {
    usage: Usage;
    history: Message[];
    contradiction: boolean;
  }) => void;
}

/**
 * SSE 版的對話。
 *
 * 用 fetch 而不是 EventSource——EventSource 只能 GET，而我們要送 history。
 *
 * 後端會先「預抽」第一段才開始串流，所以連線階段的錯誤仍然是正常的狀態碼
 * （429／503）。開始串流之後標頭已經送出去了，那之後的失敗只能靠 error 事件
 * ——不處理的話畫面會停在半截答案上。
 */
export async function chatStream(
  req: { side: Side; question: string; history: Message[] },
  handlers: StreamHandlers,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
  } catch {
    throw new ApiError(`連不上後端 ${BASE}——先跑 uvicorn api.main:app --reload`);
  }
  if (!res.ok) throw new ApiError(await detailOf(res), res.status);
  if (!res.body) throw new ApiError("後端沒有回傳串流");

  const reader = res.body.getReader();
  const decoder = new TextDecoder(); // 預設就是 utf-8
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    // stream: true 很重要——一個多位元組的中文字可能被切在兩個 chunk 中間
    buffer += decoder.decode(value, { stream: true });

    let cut: number;
    while ((cut = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      dispatch(frame, handlers);
    }
  }
}

function dispatch(frame: string, handlers: StreamHandlers) {
  let event = "";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event: ")) event = line.slice(7);
    else if (line.startsWith("data: ")) data = line.slice(6);
  }
  if (!event || !data) return;

  const payload = JSON.parse(data);
  if (event === "sources") handlers.onSources?.(payload.sources);
  else if (event === "token") handlers.onToken?.(payload.text);
  else if (event === "done") handlers.onDone?.(payload);
  else if (event === "error") throw new ApiError(payload.detail);
}
