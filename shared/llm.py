# llm.py（「跟語言模型講話」的共同介面）
#
#   為什麼要有這一層：原本雲端與地端各有一個 ChatBot，兩份幾乎一樣的
#   檢索、組 prompt、更新 history 邏輯，靠「請照著改」維持一致。當時判斷
#   它們不能合併，理由是 history 格式不同、回傳的東西也不同。
#
#   兩個理由後來都不成立：
#     · history 格式（Gemini 的 parts/model vs OpenAI 的 content/assistant）
#       是「線路格式」，屬於傳輸層。讓它決定上層架構是搞錯分層。
#     · 「地端不計費所以沒有 usage」把「不計費」和「量不到」混為一談。
#       llama.cpp 的 /v1/chat/completions 實測會回 prompt_tokens 與
#       completion_tokens——而且地端更需要看：知道 prompt 有多長，才知道
#       離 -c 的上限還有多遠（開 -c 4096 撈五個職缺會直接回 400）。
#
#   所以改成：上層只認中性格式，各家的轉換收進各自的子類。
#   加第三家（OpenRouter、OpenAI…）就是多一個檔案，不是多一個 ChatBot。
from dataclasses import dataclass


@dataclass(frozen=True)
class Usage:
    """一次生成用掉的 token。

    兩邊都拿得到，但意義不同：雲端是帳單，地端是「離 context 上限還有多遠」。
    """

    prompt: int = 0
    output: int = 0        # 雲端這個數字含思考 token——看不到但會計費

    def __add__(self, other):
        return Usage(self.prompt + other.prompt, self.output + other.output)


@dataclass(frozen=True)
class Reply:
    text: str
    usage: Usage


class BaseLLM:
    """子類要實作 complete()。

    messages 是中性格式，跟 OpenAI 的一樣但只取最小集合：
        [{"role": "user" | "assistant", "content": str}, ...]
    system 另外傳，不混在 messages 裡——Gemini 是獨立的 system_instruction 欄位，
    OpenAI 是 messages 的第一則，兩邊的塞法不同，那是子類的事。
    """

    def complete(self, messages, system, temperature=0.2) -> Reply:
        raise NotImplementedError

    def explain(self, exc):
        """把這家「可以預期會發生」的錯誤翻成一句人話，不認得就回 None。

        錯誤代碼的意思是 provider 的知識（429 對 Gemini 是配額、對別家可能是
        別的意思），所以放在這裡而不是 UI。UI 只要 explain(e) or str(e)。
        """
        return None

    def __repr__(self):
        return f"{type(self).__name__}({getattr(self, 'model', '?')})"
