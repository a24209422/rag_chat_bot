# probe_threshold.py（量 retrieve 的 min_score 該設多少）
#   用法：python -m tools.probe_threshold cloud     ← 量雲端（會打 embedding API）
#         python -m tools.probe_threshold onperm    ← 量地端（純本機，不花錢）
#   換 embedding 模型或大幅增修 DOCS 之後要重跑——門檻不是通用常數。
import sys

sys.stdout.reconfigure(encoding="utf-8")      # Windows 主控台預設 cp950，中文會變亂碼
sys.stderr.reconfigure(encoding="utf-8")      # 錯誤訊息也要，不然防呆的中文會變亂碼

SIDE = sys.argv[1] if len(sys.argv) > 1 else "cloud"
if SIDE not in ("cloud", "onperm"):
    raise SystemExit("第一個參數要是 cloud 或 onperm，收到 %r" % SIDE)

import numpy as np                            # noqa: E402

import providers                              # noqa: E402
                                              # ↑ 擺在參數檢查之後：
                                              #   打錯 side 時不必先等
                                              #   sentence_transformers 載入

print(f"量的是：{SIDE}")

GROUPS = {
    "資料裡有的（應該答得出來）": [
        "有哪些無人機相關的職缺？", "IC 設計實習生要什麼條件？",
        "耐能智慧在徵什麼人？", "有沒有可以全遠端的工作？",
        "哪個職缺需要會 Kubernetes？",
    ],
    "資料裡沒有的（應該拒答）": [
        "今天台北天氣如何？", "高鐵票價多少？", "推薦一家餐廳",
        "幫我寫一首詩", "教我怎麼煮義大利麵",
    ],
    "跟隨問句（要靠上下文才懂）": [
        "那薪水呢？", "再說詳細一點", "還有其他的嗎？",
        "那個要什麼學歷？", "代號是多少？",
    ],
}

retriever = providers.retriever_for(SIDE)     # ← 換邊只換這裡
docs = retriever.docs
D = retriever.doc_vecs()                      # 建索引（雲端會打一次 API；地端本機算）
tops = {}

for name, questions in GROUPS.items():
    print(f"\n【{name}】")
    firsts = []
    for q in questions:
        qv = retriever.embed([q], "RETRIEVAL_QUERY")[0]
        s = D @ qv
        i, j = np.argsort(-s)[:2]             # 看前兩名：第2名通常就是雜訊的高度
        firsts.append(float(s[i]))
        print(f"  {q:<18} 第1名 {s[i]:.3f}   第2名 {s[j]:.3f}   ← {docs[i].label[:10]}…")
    tops[name] = firsts
    print(f"  {'範圍':<18} {min(firsts):.3f} ～ {max(firsts):.3f}")

has_data = tops["資料裡有的（應該答得出來）"]
no_data = tops["資料裡沒有的（應該拒答）"]
followup = tops["跟隨問句（要靠上下文才懂）"]

lo, hi = max(no_data), min(has_data)          # 間隙的兩端
gap = hi - lo
print(f"\n間隙：{lo:.3f}（沒有的最高） → {hi:.3f}（有的最低），寬 {gap:.3f}")
print(f"組內全距：有的 {max(has_data) - min(has_data):.3f}、"
      f"沒有的 {max(no_data) - min(no_data):.3f}")

# 穩定性：逐一抽掉一個問句重算間隙。
#   間隙如果是由單一個問句撐著的，抽掉它就會大幅改變——那個門檻就不能信。
#   每組只有 5 個問句，任何統計都很粗；這個檢查直接量「少一個樣本會差多少」，
#   比拍一個固定的比值門檻誠實。實測地端 8.3 倍、雲端 1.4 倍。
alt = ([hi - max(no_data[:i] + no_data[i + 1:]) for i in range(len(no_data))]
       + [min(has_data[:i] + has_data[i + 1:]) - lo for i in range(len(has_data))])
swing = max(alt) / gap if gap > 0 else float("inf")
print(f"穩定性：抽掉任一個問句，間隙會變成 {min(alt):.3f} ~ {max(alt):.3f}"
      f"（原本 {gap:.3f}，最大 {swing:.1f} 倍）")
print("  ⚠ 每組只有 5 個問句，以下判定是粗估")

FLOOR = hi - 0.05        # 寬鬆下限：壓在「有的最低」之下，保證不誤殺正常問句

usable = False
if gap <= 0:
    print(f"\n判定：不可用。兩群重疊 {-gap:.3f}，單靠分數門檻分不開。")
    effective = FLOOR
elif swing >= 2:
    print(f"\n判定：不可用。間隙由單一個問句撐著——抽掉它就變 {swing:.1f} 倍，"
          f"再多一個測試問句就可能把 {gap:.3f} 填掉。")
    effective = FLOOR
else:
    # 位數要夠：%.2f 在間隙比 0.01 窄時會把中點捨到間隙外面去，
    # 反而給出一個擋不住離題的數字。所以逐步加位數直到「印出來的值」也落在間隙內。
    mid = (lo + hi) / 2
    for digits in (2, 3, 4, 5, 6):
        shown = round(mid, digits)
        if lo < shown < hi:
            break
    print(f"\n判定：可用。建議 min_score = {shown}"
          f"（間隙中點，已確認落在 {lo:.3f} ~ {hi:.3f} 之間）")
    effective, usable = shown, True

if not usable:
    print("  不要拿 min_score 當離題守門員。改成：")
    print("    · 離題判斷交給模型（shared/knowledge.py 的 SYSTEM 那句「資料沒提到的就說…」）")
    print("    · 篩選型問句用 metadata 過濾（shared/facets.py）")
    print(f"  min_score 只留寬鬆下限，建議 {FLOOR:.2f}（= 有的最低 {hi:.3f} 減 0.05）——"
          "作用是擋掉明顯無關的塊，不是擋問句。")

if max(followup) < effective:
    print(f"注意：跟隨問句最高才 {max(followup):.3f}，也會被擋掉——"
          "所以 ask() 用 `not hits and not history` 只在沒有上下文時才短路。")
