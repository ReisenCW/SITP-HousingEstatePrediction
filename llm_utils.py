from dashscope import Generation
import logging

def call_llm(model: str, prompt: str, system_prompt: str = None, search: bool = False) -> str:
    """
    封装 LLM 调用，支持 System Role 和联网搜索。
    
    Args:
        model: 模型名称 (如 "qwen-max")
        prompt: 用户提示词内容
        system_prompt: 系统人格定义
        search: 是否启用联网搜索
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
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
