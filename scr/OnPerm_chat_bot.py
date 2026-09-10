import requests
from OnPerm_rag import retrieve                                        # 1 檔案開頭

URL = "http://localhost:8080/v1/chat/completions"
messages = [{"role": "system", "content": "你是友善的中文助理，回答簡潔。"}]

while True:
    user = input("\n你 > ").strip()
    if user in ("exit", "quit", ""):
        break

    hits = retrieve(user, k=2)                                  # 2 迴圈裡，先查
    context = "\n".join(f"[{i+1}] {d}" for i, (d, _) in enumerate(hits))

    messages.append({                                           # 3 資料與問題一起送
        "role": "user",
        "content": f"【資料】\n{context}\n\n【問題】\n{user}",
    })

    if len(messages) > 11:                                      # 預防爆 context
        messages = [messages[0]] + messages[-10:]               # system 永遠保留

    try:
        resp = requests.post(URL, json={"messages": messages, "temperature": 0.7}, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"✗ 連不上 {URL}")
        print("  → llama.cpp server 沒開。WSL 裡跑：cd ~/rag-chatbot && make start_llama_server_cuda")
        messages.pop()          # 把剛剛 append 的問題收回來，不然歷史會壞掉
        continue                # 回到迴圈開頭，不要整支程式死掉
    except requests.exceptions.Timeout:
        print("✗ 等超過 120 秒，模型可能還在載入，稍等再問")
        messages.pop()
        continue

    data = resp.json()
    if "choices" not in data:
        print(f"✗ server 回了非預期內容：{data}")
        messages.pop()
        continue
    reply = data["choices"][0]["message"]["content"]

    print("AI >", reply)
    messages.append({"role": "assistant", "content": reply})
