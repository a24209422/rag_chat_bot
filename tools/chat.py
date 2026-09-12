# chat.py（互動式對話，兩邊共用）
#   用法：python -m tools.chat cloud     ← 雲端（Gemini）
#         python -m tools.chat onperm    ← 地端（需要 llama.cpp server）
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


def main(side):
    bot = providers.chat_for(side)
    history = []
    total = None

    print(f"（{side}）輸入 exit 或空白行結束")
    while True:
        try:
            user = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user in ("exit", "quit", ""):
            break

        try:
            reply, hits, usage = bot.ask(user, history)
        except Exception as e:
            print("✗", bot.llm.explain(e) or e)     # 認得的錯誤翻成人話
            continue                                # 回到迴圈，不要整支程式死掉

        for d, s in hits:                           # ← 看檢索品質
            print(f"  ↳ {s:.3f}  {d.label[:34]}…")
        total = usage if total is None else total + usage
        print(f"  [token] 入 {usage.prompt} 出 {usage.output}"
              f"（累計 {total.prompt}/{total.output}）")
        print("AI >", reply)


if __name__ == "__main__":
    side = sys.argv[1] if len(sys.argv) > 1 else "cloud"
    if side not in providers.SIDES:
        raise SystemExit("第一個參數要是 cloud 或 onperm，收到 %r" % side)
    main(side)
