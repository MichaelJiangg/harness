"""通过 Tavily Search API 搜索公开网页；API key 只从本地环境读取。"""

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from ..config import load_tavily_api_key
from .definition import ToolDefinition
from .executor import ToolError


TAVILY_ENDPOINT = "https://api.tavily.com/search"
MAX_QUERY_CHARS = 500
MAX_RESULTS = 10
MAX_RESPONSE_BYTES = 2_097_152
TIMEOUT_SECONDS = 15


DEFINITION = ToolDefinition(
    name="web_search",
    description=(
        "使用 Tavily 搜索公开网页，返回标题、URL、内容摘要和相关度。"
        "适合调研未知产品、竞品、行业动态和公开资料；max_results 默认 5，最多 10。"
        "搜索完成后应优先使用 web_fetch 读取重点 URL。"
        "需要本地配置 TAVILY_API_KEY；未配置时返回可理解的配置错误。"
        "默认需要用户确认，搜索词和结果不会写入权限日志正文。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "要搜索的完整问题或关键词。",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_RESULTS,
                "default": 5,
                "description": f"最多返回多少条结果，范围 1～{MAX_RESULTS}，默认 5。",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    supports_cancellation=True,
)


def execute(arguments, workspace, *, opener=None, api_key=None, abort=None):
    query = arguments["query"]
    max_results = arguments.get("max_results", 5)
    if not isinstance(query, str) or not query.strip() or "\x00" in query:
        raise ToolError("invalid_arguments", "query 必须是非空搜索词。")
    if len(query) > MAX_QUERY_CHARS:
        raise ToolError("invalid_arguments", "搜索词过长，请简化。")
    if type(max_results) is not int or not 1 <= max_results <= MAX_RESULTS:
        raise ToolError("invalid_arguments", f"max_results 必须是 1～{MAX_RESULTS} 之间的整数。")
    if api_key is None:
        api_key = load_tavily_api_key()
    if not api_key or not api_key.strip():
        raise ToolError(
            "web_search_unconfigured",
            "未配置 TAVILY_API_KEY；请在项目根目录 .env 中设置后重试。",
        )
    if abort is not None and abort.is_set():
        raise ToolError("execution_cancelled", "网页搜索已取消。")

    request = Request(
        TAVILY_ENDPOINT,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "query": query.strip(),
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
        }, ensure_ascii=False).encode("utf-8"),
    )
    selected_opener = urlopen if opener is None else opener
    try:
        response = selected_opener(request, timeout=TIMEOUT_SECONDS)
    except ToolError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError):
        raise ToolError("web_search_error", "无法访问 Tavily 搜索服务，请稍后重试。") from None

    with response:
        if abort is not None and abort.is_set():
            raise ToolError("execution_cancelled", "网页搜索已取消。")
        status = getattr(response, "status", None)
        if status is None:
            status = response.getcode() if hasattr(response, "getcode") else 200
        if type(status) is not int or not 200 <= status < 300:
            raise ToolError("web_search_error", f"Tavily 返回 HTTP {status}。")
        data = response.read(MAX_RESPONSE_BYTES + 1)
        if len(data) > MAX_RESPONSE_BYTES:
            raise ToolError("web_search_error", "Tavily 响应超过安全大小限制。")
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ToolError("web_search_error", "Tavily 返回了无法解析的响应。") from None
        if not isinstance(payload, dict):
            raise ToolError("web_search_error", "Tavily 返回了无效响应格式。")

    results = _clean_results(payload.get("results"))
    answer = payload.get("answer")
    return {
        "status": "success",
        "query": query.strip(),
        "answer": _clean_text(answer) if isinstance(answer, str) else "",
        "results": results,
        "returned_count": len(results),
        "message": f"搜索完成，返回 {len(results)} 条结果。",
    }


def _clean_results(value):
    if not isinstance(value, list):
        return []
    results = []
    for item in value:
        if not isinstance(item, dict):
            continue
        title = _clean_text(item.get("title")) if isinstance(item.get("title"), str) else ""
        content = _clean_text(item.get("content")) if isinstance(item.get("content"), str) else ""
        url = item.get("url")
        if (not isinstance(url, str) or not url or urlsplit(url).scheme not in {"http", "https"}
                or "\x00" in url):
            continue
        score = item.get("score")
        published_date = item.get("published_date")
        result = {
            "title": title[:300],
            "url": url[:2048],
            "content": content[:4000],
            "score": score if isinstance(score, (int, float)) else None,
            "published_date": published_date if isinstance(published_date, str) else None,
        }
        if result["title"] or result["content"]:
            results.append(result)
    return results[:MAX_RESULTS]


def _clean_text(value):
    return "".join(
        character for character in value
        if character in "\n\t" or character.isprintable()
    ).strip()
