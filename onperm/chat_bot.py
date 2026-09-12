# onperm/chat_bot.py（RAG 版對話）
import requests

from onperm.rag import OnpremRetriever
from shared.knowledge import SYSTEM

URL = "http://localhost:8080/v1/chat/completions"
HISTORY_LIMIT = 11       # 預防爆 context。奇數 → 切完第一則仍是 user，配對不會歪


class OnpremChatBot:
    """檢索 + 生成（llama.cpp server 的 OpenAI 相容 API）。

    刻意「不」跟雲端共用 base class——理由見 cloud/chat_bot.py 的 class docstring。
    """

    def __init__(self, retriever=None, url=URL, timeout=120):
        self.retriever = retriever if retriever is not None else OnpremRetriever()
        self.url = url
        self.timeout = timeout

    def ask(self, user, history, k=5):   # 5 個不同職缺。問「有哪些…」要撈得回一串，
                                         # k=2 的時代是 FAQ，一題只有一個正確答案
        """檢索 + 生成。history 會就地更新，回傳 (reply, hits)。
        沒有 usage：本機推論不計費，沒有配額可省，所以不像雲端版要數 token。
        失敗時丟例外，history 維持呼叫前的樣子。"""
        hits = self.retriever.retrieve(user, k=k)             # ← 先檢索
        if not hits and not history:  # ← 第一句就離題才短路；有上下文交給模型判斷
            reply = "資料裡沒有。"    # 措辭跟 SYSTEM 一致，兩條路說法才不會打架
            history.append({"role": "user", "content": user})
            history.append({"role": "assistant", "content": reply})
            return reply, hits        # 省掉一次本機推論——地端省的是等待，不是配額

        # 真正可靠的完整清單是 UI 的來源列（每一筆 hit 都會顯示，label 以代號開頭），
        # 模型的散文只當摘要看——實測 3B 模型不照指示抄代號，20 筆也只列 16~17 筆。
        if len(hits) > 5:      # 篩選型問句可能撈回 20 個職缺（例如「台北的職缺」），
            context = "\n".join("[%d] %s" % (i + 1, d.label)       # label 本身以代號開頭
                                for i, (d, _) in enumerate(hits))
        else:                  # 少數幾筆就給完整職缺，模型才答得出細節
            context = "\n\n".join("[%d] 代號 %s\n%s" % (i + 1, d.group, d.full)
                                  for i, (d, _) in enumerate(hits))
        prompt = f"【資料】\n{context}\n\n【問題】\n{user}"

        history.append({"role": "user", "content": user})     # 歷史存乾淨的

        if len(history) > HISTORY_LIMIT:        # 只留最近 HISTORY_LIMIT 則
            del history[:-HISTORY_LIMIT]

        to_send = list(history)                               # ← 分離送出的版本
        to_send[-1] = {"role": "user", "content": prompt}     # 只有這一輪帶【資料】
        to_send.insert(0, {"role": "system", "content": SYSTEM})  # system 不進歷史，每次現加

        try:
            resp = requests.post(self.url,
                                 json={"messages": to_send, "temperature": 0.2},
                                 timeout=self.timeout)
            resp.raise_for_status()
        except Exception:
            history.pop()      # 把剛剛 append 的問題收回來，不然歷史會壞掉
            raise

        data = resp.json()
        if "choices" not in data:
            history.pop()
            raise RuntimeError(f"server 回了非預期內容：{data}")
        reply = data["choices"][0]["message"]["content"]

        history.append({"role": "assistant", "content": reply})
        return reply, hits


if __name__ == "__main__":       # python -m onperm.chat_bot
    import sys
    sys.stdout.reconfigure(encoding="utf-8")   # Windows 主控台預設 cp950，中文會亂碼
    bot = OnpremChatBot()
    history = []

    while True:
        user = input("\n你 > ").strip()
        if user in ("exit", "quit", ""):
            break

        try:
            reply, _ = bot.ask(user, history)
        except requests.exceptions.ConnectionError:
            print(f"✗ 連不上 {bot.url}")
            print("  → llama.cpp server 沒開。另開一個終端跑（詳見 README）：")
            # noqa 理由：這是要整行複製貼上的指令，折行就不能用了
            print("     llama-server.exe -m <模型>.gguf --port 8080 -c 8192 -ngl 99 --device Vulkan0")  # noqa: E501
            continue                # 回到迴圈開頭，不要整支程式死掉
        except requests.exceptions.Timeout:
            print("✗ 等超過 120 秒，模型可能還在載入，稍等再問")
            continue
        except Exception as e:
            print(f"✗ {e}")
            continue

        print("AI >", reply)
