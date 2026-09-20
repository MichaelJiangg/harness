"""用标准库调用 OpenAI Chat Completions 兼容模型 API。"""

import json
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import get_settings

_SETTINGS = get_settings()
_MODEL = _SETTINGS["model"]
_GLM = _SETTINGS["glm"]
DEFAULT_MODEL = _MODEL["name"]


class APIError(RuntimeError):
    request_attempted = True

    def __init__(self, message, *, response=None, status_code=None, retryable=False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        if isinstance(response, dict) and isinstance(response.get("usage"), dict):
            self.response = {key: response[key] for key in ("model", "created", "usage") if key in response}


class ChatCompletionClient:
    def __init__(self, api_key, *, opener=urlopen, timeout=None, provider="deepseek", model=None):
        if provider not in {"deepseek", "glm"}:
            raise ValueError("provider 只支持 deepseek 或 glm。")
        provider_settings = _MODEL if provider == "deepseek" else _GLM
        self.provider = provider
        self.model = provider_settings["name"] if model is None else model
        self.endpoint = provider_settings["endpoint"]
        self.label = "DeepSeek" if provider == "deepseek" else "GLM"
        self.api_key_name = "DEEPSEEK_API_KEY" if provider == "deepseek" else "GLM_API_KEY"
        self.pricing = _SETTINGS["pricing"] if provider == "deepseek" else provider_settings["pricing"]
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError(f"请在项目 .env 或环境变量中设置 {self.api_key_name}。")
        self._api_key = api_key.strip()
        self._opener = opener
        self.timeout = provider_settings["request_timeout"] if timeout is None else timeout

    def complete(self, *, model, messages, tools, on_text=None, max_tokens=None):
        body = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto" if tools else "none",
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.provider == "deepseek":
            body["thinking"] = {"type": "disabled"}
        else:
            body["thinking"] = {"type": "enabled"}
            body["reasoning_effort"] = _GLM["reasoning_effort"]
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        request = Request(
            self.endpoint,
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
                return _read_stream(response, result, on_text, self.label)
        except HTTPError as error:
            hints = {
                400: "请求格式有误。", 401: f"请检查 {self.api_key_name}。",
                402: "账户余额不足。", 422: "请求参数有误。",
                429: "请求过于频繁，请稍后再试。",
                500: "服务内部错误，请稍后再试。", 503: "服务繁忙，请稍后再试。",
            }
            error.close()
            raise APIError(
                f"{self.label} 请求失败（HTTP {error.code}）。{hints.get(error.code, '')}",
                response=result, status_code=error.code,
                retryable=error.code in {429, 500, 502, 503, 504},
            ) from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise APIError(f"{self.label} 返回了无法解析的 JSON。", response=result) from None
        except TimeoutError:
            raise APIError(f"{self.label} 请求超时，请稍后再试。", response=result, retryable=True) from None
        except URLError as error:
            message = (
                f"{self.label} 请求超时。"
                if isinstance(error.reason, TimeoutError)
                else f"{self.label} 连接失败，请检查网络。"
            )
            raise APIError(message, response=result, retryable=True) from None
        except (OSError, HTTPException):
            raise APIError(f"{self.label} 连接中断，请检查网络后重试。", response=result, retryable=True) from None


DeepSeekClient = ChatCompletionClient


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


def _read_stream(response, result, on_text, provider_label="DeepSeek"):
    choice = result["choices"][0]
    message = choice["message"]
    calls = {}
    for payload in _sse_events(response):
        if payload == "[DONE]":
            if choice["finish_reason"] is None:
                raise APIError(
                    f"{provider_label} 流式回复提前结束，本轮未完成。",
                    response=result, retryable=True,
                )
            if calls:
                message["tool_calls"] = [calls[index] for index in sorted(calls)]
            return result

        chunk = json.loads(payload)
        if not isinstance(chunk, dict):
            raise APIError(f"{provider_label} 返回了无效的流式响应。", response=result)
        for key in ("id", "model", "created"):
            if chunk.get(key) is not None:
                result[key] = chunk[key]
        if chunk.get("usage") is not None:
            result["usage"] = _normalize_usage(chunk["usage"])
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            raise APIError(f"{provider_label} 返回了无效的流式响应。", response=result)
        for part in choices:
            if not isinstance(part, dict) or part.get("index") != 0:
                raise APIError(f"{provider_label} 返回了无效的流式响应。", response=result)
            delta = part.get("delta")
            if not isinstance(delta, dict):
                raise APIError(f"{provider_label} 返回了无效的流式响应。", response=result)
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise APIError(
                        f"{provider_label} 返回的文字内容格式无效。",
                        response=result,
                    )
                message["content"] += content
                if content and on_text is not None:
                    on_text(content)
            fragments = delta.get("tool_calls") or []
            if not isinstance(fragments, list):
                raise APIError(f"{provider_label} 返回的工具调用格式无效。", response=result)
            for fragment in fragments:
                if (not isinstance(fragment, dict) or type(fragment.get("index")) is not int
                        or fragment["index"] < 0):
                    raise APIError(
                        f"{provider_label} 返回的工具调用格式无效。",
                        response=result,
                    )
                call = calls.setdefault(fragment["index"], {
                    "id": "", "type": "", "function": {"name": "", "arguments": ""},
                })
                _append_fields(call, fragment, ("id", "type"), result, provider_label)
                function = fragment.get("function")
                if function is not None:
                    if not isinstance(function, dict):
                        raise APIError(
                            f"{provider_label} 返回的工具调用格式无效。",
                            response=result,
                        )
                    _append_fields(call["function"], function, ("name", "arguments"), result, provider_label)
            if part.get("finish_reason") is not None:
                choice["finish_reason"] = part["finish_reason"]
    raise APIError(
        f"{provider_label} 流式连接中断，本轮未完成，请重试。",
        response=result, retryable=True,
    )


def _append_fields(target, fragment, keys, response, provider_label):
    for key in keys:
        value = fragment.get(key)
        if value is not None:
            if not isinstance(value, str):
                raise APIError(
                    f"{provider_label} 返回的工具调用格式无效。",
                    response=response,
                )
            target[key] += value


def _normalize_usage(usage):
    if not isinstance(usage, dict):
        return usage
    normalized = dict(usage)
    details = normalized.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    prompt = normalized.get("prompt_tokens")
    if (type(cached) is int and type(prompt) is int and 0 <= cached <= prompt
            and "prompt_cache_hit_tokens" not in normalized):
        normalized["prompt_cache_hit_tokens"] = cached
        normalized.setdefault("prompt_cache_miss_tokens", prompt - cached)
    return normalized
