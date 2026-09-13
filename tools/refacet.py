# refacet.py（facets 的規則改了之後，就地重算已經存起來的分類）
#   用法：python -m tools.refacet [--check]
#
#   為什麼需要：facets 是在建置／上傳的當下算好、存進 data/jobs.json、索引檔
#   與登記簿的（見 shared/facets.py 的 derive）。規則改了不會回頭影響已經存
#   下來的東西——而基礎語料的 PDF 不在版控裡，重跑 tools.build_jobs 不見得
#   有原始檔可用。Doc 一直有留 meta（欄位原文），所以重算不需要 PDF。
#
#   ⚠ 不動向量。facets 只是 metadata，跟 embedding 無關，所以不必重算 140 個
#     向量——雲端那邊重算等於 140 次 API 呼叫再加上跨配額視窗的等待。
#
#   ⚠ 三個地方都要改。只改 jobs.json 是最容易犯的錯：索引檔存的是自己那一份
#     Doc，載得起來就不會回頭看 jobs.json（見 BaseRetriever._build_store），
#     所以跑起來的服務仍然用著舊分類，而且完全看不出來。
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from shared.facets import derive  # noqa: E402 ← 擺在 reconfigure 之後
from shared.settings import settings  # noqa: E402


def refaceted(rows):
    """就地重算每一塊的 facets。回傳有幾塊變了。"""
    changed = 0
    for r in rows:
        new = derive(r.get("meta") or {})
        if new != r.get("facets"):
            r["facets"] = new
            changed += 1
    return changed


def do_json(path, write):
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    changed = refaceted(rows)
    if changed and write:
        # 結尾的換行跟 build_jobs 一樣要補，理由見那邊。
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    return changed, len(rows)


def do_store(path, write):
    """索引檔：只換 docs 那一欄，vecs 與 meta 原封不動搬過去。"""
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    rows = json.loads(str(z["docs"]))
    changed = refaceted(rows)
    if changed and write:
        np.savez(path, vecs=z["vecs"],
                 docs=np.array(json.dumps(rows, ensure_ascii=False)),
                 meta=z["meta"])
    return changed, len(rows)


def do_registry(path, write):
    """登記簿：上傳的文件把自己的塊存成一欄 JSON。"""
    if not path.exists():
        return None
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    total = changed = 0
    try:
        rows = con.execute("SELECT doc_id, chunks FROM documents").fetchall()
        for r in rows:
            chunks = json.loads(r["chunks"])
            total += len(chunks)
            n = refaceted(chunks)
            changed += n
            if n and write:
                con.execute("UPDATE documents SET chunks = ? WHERE doc_id = ?",
                            (json.dumps(chunks, ensure_ascii=False), r["doc_id"]))
        if write:
            con.commit()
    finally:
        con.close()
    return changed, total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只報告，不寫入")
    a = ap.parse_args()
    cfg = settings()

    targets = [
        ("基礎語料 jobs.json", do_json, cfg.jobs_path),
        ("索引 cloud", do_store, cfg.cloud_store_path),
        ("索引 onperm", do_store, cfg.onperm_store_path),
        ("登記簿（上傳的文件）", do_registry, cfg.registry_path),
    ]
    for name, fn, path in targets:
        got = fn(Path(path), write=not a.check)
        if got is None:
            print("%-22s 沒有這個檔，跳過" % name)
            continue
        changed, total = got
        print("%-22s %d/%d 塊的分類變了%s"
              % (name, changed, total, "（--check，沒有寫入）" if a.check else ""))
