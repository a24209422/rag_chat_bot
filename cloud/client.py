# cloud/client.py（Gemini client 的唯一建構點）
#
#   rag.py 與 llm.py 都要一個 client。放在這裡是為了讓「讀金鑰」只有一處，
#   而且是在「呼叫時」才做，不是 import 時——模組層 genai.Client(...) 的問題是
#   沒有金鑰連 import 都會炸，於是測試跑不了、也載不進地端那側。
from google import genai

from shared.settings import settings


def make_client(api_key=None, config=None):
    """金鑰來源見 shared/settings.py：建構參數 > 環境變數 > .env。

    ⚠ settings() 有快取。Streamlit Cloud 那條路是把 secrets 塞進環境變數，
      所以那件事必須發生在第一次呼叫 settings() 之前（cloud_app.py 開頭就做）。
    """
    if api_key is None:
        api_key = (config or settings()).gemini_api_key
    if not api_key:
        raise RuntimeError(
            "找不到 GEMINI_API_KEY：本機放 .env，雲端放 App settings → Secrets")
    return genai.Client(api_key=api_key)
