"""用标准库调用 DeepSeek，避免为最小骨架引入 SDK。"""

import json
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_MODEL = "deepseek-flash"


class APIError(RuntimeError):
    request_attempted = True

    def __init__(self, message, *, response=None, status_code=None, retryable=False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        if isinstance(response, dict) and isinstance(response.get("usage"), dict):
            self.response = {key: response[key] for key in ("model", "created", "usage") if key in response}


class DeepSeekClient:
    def __init__(self, api_key, *, opener=urlopen, timeout=120):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("请在项目 .env 或环境变量中设置 DEEPSEEK_API_KEY。")
        self._api_key = api_key.strip()
        self._opener = opener
        self.timeout = timeout

    def complete(self, *, model, messages, tools, on_text=None, max_tokens=None):
        body = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto" if tools else "none",
            "thinking": {"type": "disabled"},
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        request = Request(
            "https://api.deepseek.com/chat/completions",
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        )
        result = {"choices": [{
            "index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": None,
        }]}
        try:
            with self._opener(request, timeout=self.timeout) as response:
                return _read_stream(response, result, on_text)
        except HTTPError as error:
            hints = {
                400: "请求格式有误。", 401: "请检查 DEEPSEEK_API_KEY。",
                402: "账户余额不足。", 422: "请求参数有误。",
                429: "请求过于频繁，请稍后再试。",
                500: "服务内部错误，请稍后再试。", 503: "服务繁忙，请稍后再试。",
            }
            error.close()
            raise APIError(
                f"DeepSeek 请求失败（HTTP {error.code}）。{hints.get(error.code, '')}",
                response=result, status_code=error.code,
                retryable=error.code in {429, 500, 502, 503, 504},
            ) from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise APIError("DeepSeek 返回了无法解析的 JSON。", response=result) from None
        except TimeoutError:
            raise APIError("DeepSeek 请求超时，请稍后再试。", response=result, retryable=True) from None
        except URLError as error:
            message = "DeepSeek 请求超时。" if isinstance(error.reason, TimeoutError) else "DeepSeek 连接失败，请检查网络。"
            raise APIError(message, response=result, retryable=True) from None
        except (OSError, HTTPException):
            raise APIError("DeepSeek 连接中断，请检查网络后重试。", response=result, retryable=True) from None


def _sse_events(response):
    data = []
    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        yield "\n".join(data)


def _read_stream(response, result, on_text):
    choice = result["choices"][0]
    message = choice["message"]
    calls = {}
    for payload in _sse_events(response):
        if payload == "[DONE]":
            if choice["finish_reason"] is None:
                raise APIError("DeepSeek 流式回复提前结束，本轮未完成。", response=result, retryable=True)
            if calls:
                message["tool_calls"] = [calls[index] for index in sorted(calls)]
            return result

        chunk = json.loads(payload)
        if not isinstance(chunk, dict):
            raise APIError("DeepSeek 返回了无效的流式响应。", response=result)
        for key in ("id", "model", "created", "usage"):
            if chunk.get(key) is not None:
                result[key] = chunk[key]
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            raise APIError("DeepSeek 返回了无效的流式响应。", response=result)
        for part in choices:
            if not isinstance(part, dict) or part.get("index") != 0:
                raise APIError("DeepSeek 返回了无效的流式响应。", response=result)
            delta = part.get("delta")
            if not isinstance(delta, dict):
                raise APIError("DeepSeek 返回了无效的流式响应。", response=result)
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise APIError("DeepSeek 返回的文字内容格式无效。", response=result)
                message["content"] += content
                if content and on_text is not None:
                    on_text(content)
            fragments = delta.get("tool_calls") or []
            if not isinstance(fragments, list):
                raise APIError("DeepSeek 返回的工具调用格式无效。", response=result)
            for fragment in fragments:
                if (not isinstance(fragment, dict) or type(fragment.get("index")) is not int
                        or fragment["index"] < 0):
                    raise APIError("DeepSeek 返回的工具调用格式无效。", response=result)
                call = calls.setdefault(fragment["index"], {
                    "id": "", "type": "", "function": {"name": "", "arguments": ""},
                })
                _append_fields(call, fragment, ("id", "type"), result)
                function = fragment.get("function")
                if function is not None:
                    if not isinstance(function, dict):
                        raise APIError("DeepSeek 返回的工具调用格式无效。", response=result)
                    _append_fields(call["function"], function, ("name", "arguments"), result)
            if part.get("finish_reason") is not None:
                choice["finish_reason"] = part["finish_reason"]
    raise APIError("DeepSeek 流式连接中断，本轮未完成，请重试。", response=result, retryable=True)


def _append_fields(target, fragment, keys, response):
    for key in keys:
        value = fragment.get(key)
        if value is not None:
            if not isinstance(value, str):
                raise APIError("DeepSeek 返回的工具调用格式无效。", response=response)
            target[key] += value
