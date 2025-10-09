
from openai import OpenAI
from config import Config
from evaluator import Evaluator


def test():
    client = OpenAI(
        api_key=Config.DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    prompt = f"""
                请联网查询黄浦区董家渡在2024上半年的实际房价趋势，输出格式：
                趋势：[上升/下降/持平]
                幅度描述：[如“大幅上升”“小幅下降”“基本持平”等]
                """
    response = client.chat.completions.create(
        model="qwen-long",
        messages=[{"role": "user", "content": prompt}],
        extra_body={"enable_search": True}
    )
    print(response.choices[0].message.content)

if __name__ == "__main__":
    print(Config.DASHSCOPE_API_KEY)