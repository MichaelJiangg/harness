"""读取公开网页并提取 UTF-8 文本；不提供搜索，也不执行网页脚本。"""

from html.parser import HTMLParser
import ipaddress
import re
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

from .definition import ToolDefinition
from .executor import ToolError


MAX_URL_CHARS = 2048
MAX_BYTES = 1_048_576
MAX_TEXT_CHARS = 30000
TIMEOUT_SECONDS = 10
ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {80, 443}
ALLOWED_CONTENT_TYPES = {
    "text/html", "text/plain", "text/markdown", "application/json",
    "application/xml", "application/xhtml+xml",
}


DEFINITION = ToolDefinition(
    name="web_fetch",
    description=(
        "读取一个用户明确提供的公开 http/https URL，并提取 UTF-8 文本。"
        "只接受明确的 URL，不提供搜索能力；未知产品或无法确定名称时，"
        "必须先向用户索取官方 URL，不能声称已联网搜索。"
        "拒绝私网、回环、链路本地、非标准端口、重定向和超过 1 MiB 的响应。"
        "网页 JavaScript 不会执行，返回内容仍可能包含广告或无关正文。"
        "默认需要用户确认。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "minLength": 1,
                "description": "需要读取的完整公开 http/https URL。",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
)


def execute(arguments, workspace, *, opener=None, abort=None):
    url = arguments["url"]
    if not isinstance(url, str) or not url.strip() or "\x00" in url:
        raise ToolError("invalid_arguments", "url 必须是非空公开网页地址。")
    if len(url) > MAX_URL_CHARS:
        raise ToolError("invalid_arguments", "url 过长，请提供简短地址。")
    parsed = urlsplit(url.strip())
    if parsed.scheme.lower() not in ALLOWED_SCHEMES or parsed.username or parsed.password:
        raise ToolError("invalid_url", "只允许不含认证信息的公开 http/https URL。")
    hostname, port = _validated_host_port(parsed)
    if abort is not None and abort.is_set():
        raise ToolError("execution_cancelled", "网页读取已取消。")

    selected_opener = opener
    if selected_opener is None:
        selected_opener = build_opener(NoRedirectHandler(), ProxyHandler({}))
    try:
        response = selected_opener(url, timeout=TIMEOUT_SECONDS)
    except ToolError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError):
        raise ToolError("fetch_error", "无法访问该网页，请检查 URL 或稍后重试。") from None

    with response:
        if abort is not None and abort.is_set():
            raise ToolError("execution_cancelled", "网页读取已取消。")
        status = getattr(response, "status", None)
        if status is None:
            status = response.getcode() if hasattr(response, "getcode") else 200
        if type(status) is not int or not 200 <= status < 300:
            raise ToolError("http_error", f"网页返回 HTTP {status}，未读取正文。")
        headers = response.headers if hasattr(response, "headers") else {}
        content_type = headers.get("Content-Type", "text/html")
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type not in ALLOWED_CONTENT_TYPES and not (
            media_type.endswith("+json") or media_type.endswith("+xml")
        ):
            raise ToolError("unsupported_content", "仅支持文本、HTML、JSON 和 XML 网页内容。")
        length = headers.get("Content-Length")
        if isinstance(length, str) and length.isdigit() and int(length) > MAX_BYTES:
            raise ToolError("too_large", f"网页超过 {MAX_BYTES} 字节，未读取。")
        data = response.read(MAX_BYTES + 1)
        if abort is not None and abort.is_set():
            raise ToolError("execution_cancelled", "网页读取已取消。")
        if len(data) > MAX_BYTES:
            raise ToolError("too_large", f"网页超过 {MAX_BYTES} 字节，已停止读取。")
        charset = getattr(headers, "get_content_charset", lambda: None)()
        text = data.decode(charset or "utf-8", errors="replace")
        final_url = response.geturl() if hasattr(response, "geturl") else url

    title, body = _extract_text(text, media_type)
    truncated = len(body) > MAX_TEXT_CHARS
    body = body[:MAX_TEXT_CHARS]
    return {
        "status": "success",
        "url": url,
        "final_url": final_url,
        "title": title,
        "text": body,
        "content_type": media_type,
        "truncated": truncated,
        "bytes_received": len(data),
        "message": f"已读取网页（{len(body)} 字符）。",
    }


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ToolError("redirect_not_supported", "网页发生重定向，已拒绝跟踪。")


def _validated_host_port(parsed):
    hostname = parsed.hostname
    if not hostname:
        raise ToolError("invalid_url", "URL 缺少有效主机名。")
    hostname = hostname.rstrip(".").lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        raise ToolError("unsafe_url", "不允许访问本机、局域网或本地域名。")
    try:
        encoded = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        raise ToolError("invalid_url", "URL 主机名无法解析。") from None
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme.lower() == "https" else 80
    if port not in ALLOWED_PORTS:
        raise ToolError("unsafe_url", "只允许使用 80 和 443 端口。")
    try:
        if isinstance(ipaddress.ip_address(encoded), ipaddress.IPv6Address):
            addresses = [(socket.AF_INET6, ipaddress.ip_address(encoded).compressed, 0, 0)]
        else:
            addresses = [(socket.AF_INET, encoded, 0, 0)]
    except ValueError:
        try:
            addresses = socket.getaddrinfo(encoded, port, type=socket.SOCK_STREAM)
        except socket.gaierror:
            raise ToolError("dns_error", "无法解析网页域名。") from None
    for address in addresses:
        try:
            candidate = ipaddress.ip_address(address[4][0])
        except (ValueError, IndexError):
            raise ToolError("unsafe_url", "域名解析结果无效，已拒绝访问。") from None
        if (candidate.is_private or candidate.is_loopback or candidate.is_link_local
                or candidate.is_multicast or candidate.is_reserved or candidate.is_unspecified):
            raise ToolError("unsafe_url", "目标地址属于私网或保留地址，已拒绝访问。")
    return encoded, port


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.title_parts = []
        self.in_title = False
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "template", "svg"}:
            self.skip_depth += 1
        if tag == "title":
            self.in_title = True
        if tag in {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3",
                   "h4", "h5", "h6", "blockquote", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "template", "svg"} and self.skip_depth:
            self.skip_depth -= 1
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)
        if self.in_title:
            self.title_parts.append(data)


def _extract_text(source, media_type):
    if media_type != "text/html":
        return "", _normalize_text(source)
    parser = _TextExtractor()
    try:
        parser.feed(source)
        parser.close()
    except Exception:
        return "", _normalize_text(source)
    title = _normalize_text(" ".join(parser.title_parts))[:300]
    body = _normalize_text("".join(parser.parts))
    return title, body


def _normalize_text(value):
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()
