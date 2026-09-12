# settings.py（所有會因環境而異的設定，集中在這裡）
#
#   在這之前，模型名、server 位址、逾時、溫度、k 散在六個檔案的模組層。
#   想換一個生成模型要翻檔案；想在測試裡改逾時只能 monkeypatch。
#
#   ⚠ min_score 刻意「不在」這裡。
#     它不是可以調的旋鈕，是 tools/probe_threshold.py 對「這個 embedding 模型
#     ＋這批語料」量出來的結果，換任一邊都要重量。放進 .env 會讓它看起來像
#     可以隨手調的參數，那正是最危險的誤解——所以它留在各自的 Retriever
#     類別上，跟量測過程的說明放在一起。要臨時試別的值就傳建構子參數：
#     retriever_for("cloud", min_score=0.7)。
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,     # .env 裡寫 FOO= 當作沒設，而不是設成空字串
        extra="ignore",            # .env 裡有別的東西不要炸
    )

    # ── 金鑰 ─────────────────────────────────────────────────────────
    # 讀取順序是 pydantic-settings 決定的：建構參數 > 環境變數 > .env > 預設值。
    # Streamlit Cloud 沒有 .env，但 cloud_app.py 會把 secrets 塞進環境變數，
    # 所以那條路也走得通。
    gemini_api_key: str = ""

    # ── 雲端：Gemini ─────────────────────────────────────────────────
    gemini_embed_model: str = "gemini-embedding-001"
    gemini_embed_dim: int = 768      # 預設 3072；截短成 768 省記憶體與比對時間（靠 MRL）
    gemini_embed_batch: int = 100    # BatchEmbedContentsRequest 一次最多 100 筆
    gemini_chat_model: str = "gemini-flash-latest"

    # ── 地端：llama.cpp + e5 ─────────────────────────────────────────
    llama_url: str = "http://localhost:8080/v1/chat/completions"
    llama_timeout: int = 120
    onperm_embed_model: str = "intfloat/multilingual-e5-small"
    # 要跟模型對得上。e5-small 是 384；換模型忘了改的話，第一次寫入索引就會
    # 當場報錯（NumpyStore.add 會檢查），不會默默算出爛向量。
    onperm_embed_dim: int = 384

    # ── 生成 ─────────────────────────────────────────────────────────
    temperature: float = 0.2
    retrieve_k: int = 5              # 5 個不同職缺。k=2 的時代是 FAQ，一題一個答案

    # history 上限：超過就從前面砍，預防爆 context。
    # 上限用奇數——切完第一則仍然是 user，問答配對不會歪掉。
    # 雲端是 None（不砍）：Gemini 的 context 寬裕得多，砍了反而失去上下文。
    cloud_history_limit: int | None = None
    onperm_history_limit: int | None = 11

    # ── API ──────────────────────────────────────────────────────────
    # Streamlit 與 CLI 都是打這個位址，不再直接 import ChatBot。
    api_url: str = "http://localhost:8000"
    api_timeout: int = 180        # 地端生成可能很慢，要比 llama_timeout 寬

    # 啟動時要先建好索引的邊，逗號分隔（例如 "cloud" 或 "cloud,onperm"）。
    # 預設空的＝全部延遲到第一次請求。用逗號字串而不是 list 型別，是因為
    # pydantic-settings 的 list 預設要求 JSON 格式（API_WARM=["cloud"]），
    # 在 .env 裡寫起來很彆扭。
    api_warm: str = ""

    # 瀏覽器前端的來源。Streamlit 與 CLI 是從伺服器端打 API，不受 CORS 管；
    # React 是從瀏覽器打，不設這個就會被擋下來（而且錯誤只出現在 devtools）。
    # 逗號分隔。* 會連帶關掉帶 cookie 的請求，所以老實列出來。
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ── 上傳的文件 ───────────────────────────────────────────────────
    max_upload_bytes: int = 10 * 1024 * 1024
    allowed_upload_suffixes: str = ".pdf"      # 逗號分隔，理由見 shared/ingest.py

    # ── 路徑 ─────────────────────────────────────────────────────────
    jobs_path: Path = ROOT / "data" / "jobs.json"       # 建置階段的基礎語料
    registry_path: Path = ROOT / "data" / "registry.db"  # 上傳文件的登記簿
    # 索引檔一邊一個——兩邊的向量空間不同，不能混
    cloud_store_path: Path = ROOT / "data" / "store_cloud.npz"
    onperm_store_path: Path = ROOT / "data" / "store_onperm.npz"

    def warm_sides(self):
        return [s.strip() for s in self.api_warm.split(",") if s.strip()]

    def cors_list(self):
        return [s.strip() for s in self.cors_origins.split(",") if s.strip()]

    def upload_suffixes(self):
        return [s.strip().lower() for s in self.allowed_upload_suffixes.split(",")
                if s.strip()]

    def store_path_for(self, side):
        return self.cloud_store_path if side == "cloud" else self.onperm_store_path


@lru_cache(maxsize=1)
def settings() -> Settings:
    """預設設定，整個 process 共用一份。

    跟 knowledge.default_docs() 一樣是「函式 + 快取」而不是模組層的實例：
    import 這個模組不會去讀 .env，測試也能自己建一個 Settings(...) 傳進去，
    不必動環境變數。
    """
    return Settings()
