# cloud/client.py（Gemini client 的唯一建構點）
#
#   rag.py 與 chat_bot.py 都要一個 client。放在這裡是為了讓「讀金鑰」這件事
#   只有一處——而且是在「呼叫時」才做，不是 import 時。
#   模組層 client = genai.Client(...) 的問題：沒有金鑰就連 import 都會炸，
#   於是測試跑不了、CI 跑不了、想同時載入地端那側也會被這行擋住。
import os

from dotenv import load_dotenv
from google import genai


def make_client(api_key=None):
    if api_key is None:
        load_dotenv()
        # ← 不要用 os.environ[...]：那樣拿不到就是 KeyError，
        #   訊息只有一個變數名，看不出該去哪裡設定
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "找不到 GEMINI_API_KEY：本機放 .env，雲端放 App settings → Secrets")
    return genai.Client(api_key=api_key)
