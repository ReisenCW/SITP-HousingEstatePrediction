from dashscope import Generation
import logging
from typing import Generator, Union

def call_llm(model: str, prompt: str, system_prompt: str = None, search: bool = False, stream: bool = False, max_tokens: int = None) -> Union[str, Generator[str, None, None]]:
    """
    封装 LLM 调用，支持 System Role、联网搜索、流式输出和字数限制。
    
    Args:
        model: 模型名称 (如 "qwen-max")
        prompt: 用户提示词内容
        system_prompt: 系统人格定义
        search: 是否启用联网搜索
        stream: 是否启用流式输出
        max_tokens: 最大生成 token 数
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # 准备调用参数
    call_params = {
        "model": model,
        "messages": messages,
        "enable_search": search,
        "result_format": "message",
        "stream": stream,
        "incremental_output": stream
    }
    if max_tokens:
        call_params["max_tokens"] = max_tokens
    
    try:
        responses = Generation.call(**call_params)
        
        if not stream:
            if responses.status_code == 200:
                return responses.output.choices[0].message.content
            else:
                logging.error(f"Error calling LLM: {responses.code} - {responses.message}")
                return ""
        else:
            def stream_generator():
                full_content = ""
                for response in responses:
                    if response.status_code == 200:
                        chunk = response.output.choices[0].message.content
                        yield chunk
                    else:
                        logging.error(f"Stream Error: {response.code} - {response.message}")
                return
            return stream_generator()

    except Exception as e:
        logging.error(f"Exception during LLM call: {e}")
        return ""
