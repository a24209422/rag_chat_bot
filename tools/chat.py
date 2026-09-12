# chat.py（互動式對話，兩邊共用）
#   用法：python -m tools.chat cloud            ← 雲端（Gemini）
#         python -m tools.chat onperm           ← 地端（需要 llama.cpp server）
#         python -m tools.chat onperm --api     ← 改成打後端 API
#
#   預設是「直接呼叫」而不是打 API，刻意的：這樣後端掛掉時還有一條路可以
#   驗證核心邏輯，HTTP 不會擋在中間。--api 則是用來快速確認跑起來的後端。
#
#   取代原本 cloud/chat_bot.py 與 onperm/chat_bot.py 各自的 __main__ 區塊。
#   那兩份幾乎一樣，差在錯誤訊息怎麼寫——現在那件事由 llm.explain() 負責。
import sys

sys.stdout.reconfigure(encoding="utf-8")   # Windows 主控台預設 cp950，中文會亂碼
sys.stderr.reconfigure(encoding="utf-8")
# stdin 也要。互動輸入走的是 Windows 的 console API（本來就 UTF-8），
# 但「管道」輸入是用 locale 解碼的——不設的話中文會被 cp950 解錯，
# 錯誤還不會發生在這裡，而是後面餵進 tokenizer 時炸一個看不懂的訊息。
try:
    sys.stdin.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):       # stdin 被換成非文字串流時
    pass

import providers  # noqa: E402 ← 擺在 reconfigure 之後


def main(side, use_api=False):
    if use_api:
        from shared.api_client import ApiClient, ApiError
        client = ApiClient()
        ask, known_errors = _via_api(client, side), ApiError
    else:
        bot = providers.chat_for(side)
        ask, known_errors = _direct(bot), Exception

    history = []
    total = None

    print(f"（{side}{'，經由 API' if use_api else ''}）輸入 exit 或空白行結束")
    while True:
        try:
            user = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user in ("exit", "quit", ""):
            break

        try:
            reply, sources, usage, history = ask(user, history)
        except known_errors as e:
            print("✗", e)
            continue                                # 回到迴圈，不要整支程式死掉

        for s in sources:                           # ← 看檢索品質
            print(f"  ↳ {s['score']:.3f}  {s['label'][:34]}…")
        total = usage if total is None else total + usage
        print(f"  [token] 入 {usage.prompt} 出 {usage.output}"
              f"（累計 {total.prompt}/{total.output}）")
        print("AI >", reply)


def _direct(bot):
    """直接呼叫。history 會被就地更新，所以原樣傳回去。"""
    def ask(user, history):
        reply, hits, usage = bot.ask(user, history)
        sources = [{"code": d.group, "label": d.label, "score": s} for d, s in hits]
        return reply, sources, usage, history
    return ask


def _via_api(client, side):
    """打後端。後端是無狀態的，history 進去什麼樣、回來就是更新後的樣子。"""
    def ask(user, history):
        a = client.ask(side, user, history)
        return a.reply, a.sources, a.usage, a.history
    return ask


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    side = args[0] if args else "cloud"
    if side not in providers.SIDES:
        raise SystemExit("第一個參數要是 cloud 或 onperm，收到 %r" % side)
    main(side, use_api="--api" in sys.argv)
