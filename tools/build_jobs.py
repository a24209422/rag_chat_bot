# build_jobs.py（把一整個資料夾的職缺 PDF 解析成 data/jobs.json）
#   用法：python -m tools.build_jobs <PDF資料夾> [-o data/jobs.json]
#
#   產出的是「基礎語料」——建置階段就固定下來的那批。執行期上傳的文件走
#   另一條路（POST /documents），不會寫進這個檔案。
#
#   解析邏輯本身在 shared/ingest.py：上傳 API 也要用同一份，兩邊才不會
#   切出形狀不同的塊。
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from shared.ingest import docs_from_pdf  # noqa: E402 ← 擺在 reconfigure 之後


def build(pdf_dir):
    """資料夾結構是每家公司一個子資料夾、裡面一份 PDF。"""
    docs, jobs = [], 0
    for p in sorted(Path(pdf_dir).glob("*/*.pdf")):
        part, n = docs_from_pdf(p.read_bytes())
        docs += part
        jobs += n
    return docs, jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_dir")
    ap.add_argument("-o", "--out", default="data/jobs.json")
    a = ap.parse_args()

    docs, jobs = build(a.pdf_dir)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    # 結尾要有換行：pre-commit 的 end-of-file-fixer 會補上，產生器不補的話
    # 每次重建 jobs.json 都會跟 hook 來回打架。
    Path(a.out).write_text(json.dumps(docs, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    print("%d 個職缺 → %d 塊 → %s" % (jobs, len(docs), a.out))
