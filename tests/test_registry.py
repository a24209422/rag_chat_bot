"""shared/registry.py：上傳文件的登記簿。"""
import pytest

from shared.registry import Registry, doc_id_for, registry_at


@pytest.fixture
def reg(tmp_path):
    return Registry(tmp_path / "reg.db")


CHUNKS = [{"id": "A-01#基本", "text": "塊一", "group": "A-01"},
          {"id": "A-01#工作內容", "text": "塊二", "group": "A-01"}]


def test_放進去再拿出來(reg):
    doc_id = reg.put("甲公司.pdf", "application/pdf", 1234, "v1", CHUNKS, jobs=1)

    row = reg.get(doc_id)
    assert row["filename"] == "甲公司.pdf"
    assert row["version"] == "v1"
    assert row["jobs"] == 1
    assert row["chunks"] == CHUNKS


def test_doc_id由檔名決定(reg):
    """同一個檔名就是同一份文件，所以「同名但內容不同」是更新而不是新增。"""
    assert reg.put("甲.pdf", "application/pdf", 1, "v1", CHUNKS, 1) == doc_id_for("甲.pdf")


def test_同名重傳是更新不是新增(reg):
    reg.put("甲.pdf", "application/pdf", 100, "v1", CHUNKS, 1)
    reg.put("甲.pdf", "application/pdf", 200, "v2", CHUNKS[:1], 1)

    assert len(reg.list()) == 1
    row = reg.get(doc_id_for("甲.pdf"))
    assert row["version"] == "v2" and row["size"] == 200
    assert len(row["chunks"]) == 1          # 舊的塊被換掉了


def test_不同檔名是兩份(reg):
    reg.put("甲.pdf", "application/pdf", 1, "v1", CHUNKS, 1)
    reg.put("乙.pdf", "application/pdf", 1, "v2", CHUNKS, 1)
    assert len(reg.list()) == 2


def test_列表不帶chunks(reg):
    """那欄很大（整份文件的塊），列表用不到。"""
    reg.put("甲.pdf", "application/pdf", 1, "v1", CHUNKS, 1)
    assert "chunks" not in reg.list()[0]


def test_刪除(reg):
    doc_id = reg.put("甲.pdf", "application/pdf", 1, "v1", CHUNKS, 1)

    assert reg.delete(doc_id)
    assert reg.get(doc_id) is None
    assert not reg.delete(doc_id)           # 刪第二次是 False，不是例外


def test_一次撈多份的chunks(reg):
    """建索引時要用。一次撈一批，不要一份一份查。"""
    a = reg.put("甲.pdf", "application/pdf", 1, "v1", CHUNKS, 1)
    b = reg.put("乙.pdf", "application/pdf", 1, "v2", CHUNKS[:1], 1)
    reg.put("丙.pdf", "application/pdf", 1, "v3", CHUNKS, 1)

    got = reg.chunks_of({a, b})
    assert set(got) == {a, b}
    assert len(got[b]) == 1


def test_沒有要撈就不要打資料庫(reg):
    assert reg.chunks_of(set()) == {}


def test_doc_ids(reg):
    reg.put("甲.pdf", "application/pdf", 1, "v1", CHUNKS, 1)
    assert reg.doc_ids() == {doc_id_for("甲.pdf")}


def test_重開還在(tmp_path):
    path = tmp_path / "reg.db"
    Registry(path).put("甲.pdf", "application/pdf", 1, "v1", CHUNKS, 1)
    assert len(Registry(path).list()) == 1


def test_registry_at會共用同一個實例(tmp_path):
    """雲端與地端的索引分開，但「上傳了哪些文件」只有一份。"""
    path = tmp_path / "reg.db"
    assert registry_at(path) is registry_at(path)
