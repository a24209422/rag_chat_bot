# registry.py（上傳文件的登記簿）
#
#   記的是「哪些文件被上傳過、各自切出哪些塊」。切好的塊存在這裡而不是只存
#   向量，有兩個理由：
#
#   1. 雲端與地端的向量空間不同，各有各的索引檔。同一份上傳的文件必須能讓
#      兩邊各自算出自己的向量——所以塊要存一份共用的。
#   2. 刪除要精準。有了 doc_id → 塊 的對照，刪一份文件就是刪它的塊，
#      不必掃整個索引，也不會誤傷別人。
#
#   用 sqlite3（標準函式庫，不加依賴）。這個規模不需要 ORM，
#   而且 SQL 直接寫出來，資料長什麼樣一目了然。
import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from functools import lru_cache

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size         INTEGER NOT NULL,
    version      TEXT NOT NULL,
    jobs         INTEGER NOT NULL,
    chunks       TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
"""


def doc_id_for(filename):
    """同一個檔名就是同一份文件。

    所以「同名但內容不同」是更新（換掉舊的塊），不是新增一份——
    這正是增量更新要處理的情況。
    """
    return hashlib.sha1(filename.encode("utf-8")).hexdigest()[:16]


class Registry:
    def __init__(self, path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def put(self, filename, content_type, size, version, chunks, jobs):
        """新增或就地更新。回傳 doc_id。"""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents "
                "(doc_id, filename, content_type, size, version, jobs, chunks,"
                " created_at) VALUES (?,?,?,?,?,?,?,?)",
                (doc_id_for(filename), filename, content_type, size, version, jobs,
                 json.dumps(chunks, ensure_ascii=False),
                 datetime.now(UTC).isoformat(timespec="seconds")))
        return doc_id_for(filename)

    def get(self, doc_id):
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id=?",
                               (doc_id,)).fetchone()
        return _row(row) if row else None

    def list(self):
        """列出所有文件，**不含** chunks——那欄很大，列表用不到。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT doc_id, filename, content_type, size, version, jobs,"
                " created_at FROM documents ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def list_one(self, doc_id):
        """單一份文件的摘要（不含 chunks）。上傳完要回給呼叫端。"""
        row = self.get(doc_id)
        return _summary(row) if row else None

    def delete(self, doc_id):
        with self._conn() as conn:
            return conn.execute("DELETE FROM documents WHERE doc_id=?",
                                (doc_id,)).rowcount > 0

    def doc_ids(self):
        with self._conn() as conn:
            return {r[0] for r in conn.execute("SELECT doc_id FROM documents")}

    def chunks_of(self, doc_ids):
        """要建索引時才撈 chunks，一次撈一批。"""
        if not doc_ids:
            return {}
        marks = ",".join("?" * len(doc_ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT doc_id, chunks FROM documents WHERE doc_id IN ({marks})",
                tuple(doc_ids)).fetchall()
        return {r["doc_id"]: json.loads(r["chunks"]) for r in rows}


def _row(row):
    d = dict(row)
    d["chunks"] = json.loads(d["chunks"])
    return d


@lru_cache(maxsize=4)
def registry_at(path):
    """同一個檔案共用一個 Registry 實例。

    雲端與地端的索引是分開的，但「上傳了哪些文件」只有一份——
    同一份文件要讓兩邊各自算自己的向量，所以塊必須是共用的。
    """
    return Registry(path)


def _summary(row):
    d = dict(row)
    d.pop("chunks", None)
    return d
