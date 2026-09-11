# OnPerm_chat_bot.py（RAG 版）
import requests
from OnPerm_rag import retrieve

URL = "http://localhost:8080/v1/chat/completions"
SYSTEM = ("你是客服助理。只依據提供的【資料】回答，"     # ← 比照 cloud_chat_bot.py：
          "資料沒提到的就說「資料裡沒有」。")           # 不講這句，模型會無視資料瞎掰


def ask(user, history, k=2):
    """檢索 + 生成。history 會就地更新，回傳 (reply, hits)。
    沒有 usage：本機推論不計費，沒有配額可省，所以不像雲端版要數 token。
    失敗時丟例外，history 維持呼叫前的樣子。"""
    hits = retrieve(user, k=k)                            # ← 先檢索
    if not hits and not history:      # ← 第一句就離題才短路；有上下文交給模型判斷
        reply = "資料裡沒有。"        # 措辭跟 SYSTEM 一致，兩條路說法才不會打架
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": reply})
        return reply, hits            # 省掉一次本機推論——地端省的是等待，不是配額

    context = "\n".join(f"[{i+1}] {d}" for i, (d, _) in enumerate(hits))
    prompt = f"【資料】\n{context}\n\n【問題】\n{user}"

    history.append({"role": "user", "content": user})     # 歷史存乾淨的

    if len(history) > 11:                  # 預防爆 context：只留最近 11 則
        del history[:-11]                  # 奇數 → 切完第一則仍是 user，配對不會歪

    to_send = list(history)                               # ← 分離送出的版本
    to_send[-1] = {"role": "user", "content": prompt}     # 只有這一輪帶【資料】
    to_send.insert(0, {"role": "system", "content": SYSTEM})   # system 不進歷史，每次現加

    try:
        resp = requests.post(URL, json={"messages": to_send, "temperature": 0.7},
                             timeout=120)
        resp.raise_for_status()
    except Exception:
        history.pop()          # 把剛剛 append 的問題收回來，不然歷史會壞掉
        raise

    data = resp.json()
    if "choices" not in data:
        history.pop()
        raise RuntimeError(f"server 回了非預期內容：{data}")
    reply = data["choices"][0]["message"]["content"]

    history.append({"role": "assistant", "content": reply})
    return reply, hits


if __name__ == "__main__":
    history = []

    while True:
        user = input("\n你 > ").strip()
        if user in ("exit", "quit", ""):
            break

        try:
            reply, _ = ask(user, history)
        except requests.exceptions.ConnectionError:
            print(f"✗ 連不上 {URL}")
            print("  → llama.cpp server 沒開。WSL 裡跑：cd ~/rag-chatbot && make start_llama_server_cuda")
            continue                # 回到迴圈開頭，不要整支程式死掉
        except requests.exceptions.Timeout:
            print("✗ 等超過 120 秒，模型可能還在載入，稍等再問")
            continue
        except Exception as e:
            print(f"✗ {e}")
            continue

        print("AI >", reply)
