"""通过 Tavily Search API 搜索公开网页；API key 只从本地环境读取。"""

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from ..config import get_settings, load_tavily_api_key
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
        "结果按当前字符预算裁剪摘要或省略整条来源，保留完整链接并标记裁剪。"
        "需要本地配置 TAVILY_API_KEY；未配置时返回可理解的配置错误。"
        "默认无需用户确认；显式 ask/deny 规则仍生效，搜索词和结果不会写入权限日志正文。"
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

    from ..orchestration import get_current_query

    state = get_current_query()
    budget = state.tool_result_limit if state is not None else get_settings()["context"]["tool_result_chars"]
    # 最小搜索信息无法回传时不发起搜索，避免执行后只能伪造空结果。
    _budget_results(query.strip(), [], "", 0, budget)

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

    results, omitted = _clean_results(payload.get("results"), max_results)
    answer = payload.get("answer")
    answer = _clean_text(answer) if isinstance(answer, str) else ""
    return _budget_results(query.strip(), results, answer, omitted, budget)


def _clean_results(value, max_results):
    if not isinstance(value, list):
        return [], 0
    results = []
    omitted = 0
    for item in value:
        if not isinstance(item, dict):
            continue
        title = _clean_text(item.get("title")) if isinstance(item.get("title"), str) else ""
        content = _clean_text(item.get("content")) if isinstance(item.get("content"), str) else ""
        url = item.get("url")
        if (not isinstance(url, str) or not url or urlsplit(url).scheme not in {"http", "https"}
                or "\x00" in url):
            continue
        if len(url) > 2048:
            omitted += 1
            continue
        score = item.get("score")
        published_date = item.get("published_date")
        result = {
            "title": title,
            "url": url,
            "content": content,
            "score": score if isinstance(score, (int, float)) else None,
            "published_date": published_date if isinstance(published_date, str) else None,
        }
        if result["title"] or result["content"]:
            results.append(result)
    return results[:max_results], omitted + max(0, len(results) - max_results)


def _budget_results(query, results, answer, omitted, budget):
    """完整来源先占预算，再按实际 JSON 成本均分可保留的摘要长度。"""
    if type(budget) is not int or budget <= 0:
        raise ToolError("search_budget_too_small", "工具结果预算无效，无法返回搜索结果。")
    retained = []

    def build(chars):
        items = [{**item, "content": item["content"][:chars],
                  "content_truncated": len(item["content"]) > chars} for item in retained]
        return {
            "status": "success", "query": query, "answer": answer[:chars],
            "answer_truncated": len(answer) > chars,
            "results": items, "returned_count": len(items), "omitted_count": omitted,
            "truncated": bool(omitted or len(answer) > chars
                              or any(item["content_truncated"] for item in items)),
            "message": (f"搜索完成，返回 {len(items)} 条结果。"
                        + (f" 已省略 {omitted} 条来源。" if omitted else "")),
        }

    def fits(result):
        complete = {**result, "executed": True, "tool": "web_search"}
        return len(json.dumps(complete, ensure_ascii=False)) <= budget

    for item in results:
        retained.append(item)
        if not fits(build(0)):
            retained.pop()
            omitted += 1

    while not fits(build(0)):
        if not retained:
            raise ToolError("search_budget_too_small", "工具结果预算不足，无法容纳搜索信息。")
        retained.pop()
        omitted += 1

    low, high = 0, max([len(answer)] + [len(item["content"]) for item in retained])
    while low < high:
        middle = (low + high + 1) // 2
        if fits(build(middle)):
            low = middle
        else:
            high = middle - 1
    return build(low)


def _clean_text(value):
    return "".join(
        character for character in value
        if character in "\n\t" or character.isprintable()
    ).strip()
