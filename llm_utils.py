from dashscope import Generation
import logging

def call_llm(model: str, prompt: str, search: bool = False) -> str:
    """
    封装 LLM 调用，支持联网搜索。
    
    Args:
        model: 模型名称 (如 "qwen-max")
        prompt: 提示词内容
        search: 是否启用联网搜索 (DashScope 的 enable_search 参数)
        
    Returns:
        LLM 返回的文本内容，失败则返回空字符串。
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        response = Generation.call(
            model=model,
            messages=messages,
            enable_search=search,
            result_format="message"
        )
        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            logging.error(f"Error calling LLM: {response.code} - {response.message}")
            return ""
    except Exception as e:
        logging.error(f"Exception during LLM call: {e}")
        return ""
