# probe_threshold.py（量 retrieve 的 min_score 該設多少）
#   用法：python scr/probe_threshold.py
#   換 embedding 模型或大幅增修 DOCS 之後要重跑——門檻不是通用常數。
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")      # Windows 主控台預設 cp950，中文會變亂碼

import numpy as np                            # noqa: E402
import cloud_rag as rag                       # noqa: E402

GROUPS = {
    "資料裡有的（應該答得出來）": [
        "退款要幾天內申請？", "運費多少錢？", "客服幾點上班？",
        "金卡會員怎麼升級？", "電子產品保固多久？",
    ],
    "資料裡沒有的（應該拒答）": [
        "今天台北天氣如何？", "你會寫 Python 嗎？", "推薦一家餐廳",
        "幫我寫一首詩", "高鐵票價多少？",
    ],
    "跟隨問句（要靠上下文才懂）": [
        "那需要什麼條件？", "再說詳細一點", "為什麼？", "還有呢？", "那超過的話呢？",
    ],
}

D = rag.doc_vecs()                            # 建索引（會打一次 embedding API）
tops = {}

for name, questions in GROUPS.items():
    print(f"\n【{name}】")
    firsts = []
    for q in questions:
        qv = rag.embed([q], "RETRIEVAL_QUERY")[0]
        s = D @ qv
        i, j = np.argsort(-s)[:2]             # 看前兩名：第2名通常就是雜訊的高度
        firsts.append(float(s[i]))
        print(f"  {q:<18} 第1名 {s[i]:.3f}   第2名 {s[j]:.3f}   ← {rag.DOCS[i][:10]}…")
    tops[name] = firsts
    print(f"  {'範圍':<18} {min(firsts):.3f} ～ {max(firsts):.3f}")

has_data = tops["資料裡有的（應該答得出來）"]
no_data = tops["資料裡沒有的（應該拒答）"]
followup = tops["跟隨問句（要靠上下文才懂）"]

gap = min(has_data) - max(no_data)
print(f"\n間隙：{max(no_data):.3f}（沒有的最高） → {min(has_data):.3f}（有的最低），寬 {gap:.3f}")
if gap <= 0:
    print("兩群重疊，單靠分數門檻分不開——別用 min_score，或換個檢索方式。")
else:
    suggest = (min(has_data) + max(no_data)) / 2
    print(f"建議門檻取中間：min_score = {suggest:.2f}")
    if max(followup) < suggest:
        print(f"注意：跟隨問句最高才 {max(followup):.3f}，也會被擋掉——"
              "所以 ask() 用 `not hits and not history` 只在沒有上下文時才短路。")
