"""读取公开网页并提取 UTF-8 文本；不提供搜索，也不执行网页脚本。"""

from contextlib import contextmanager, ExitStack
from html.parser import HTMLParser
from http.client import HTTPConnection, HTTPException
import ipaddress
import json
import re
import socket
import ssl
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit

from .definition import ToolDefinition
from .executor import ToolError


MAX_URL_CHARS = 2048
MAX_BYTES = 1_048_576
MAX_TEXT_CHARS = 30000
TIMEOUT_SECONDS = 10
MAX_DNS_BYTES = 65536
FAKE_IP_NETWORK = ipaddress.ip_network("198.18.0.0/15")
IPV6_TRANSLATION_NETWORKS = (
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/48"),
)
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
        "默认无需用户确认；显式 ask/deny 规则和上述安全检查仍生效。"
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
    supports_cancellation=True,
)


def execute(arguments, workspace, *, opener=None, abort=None):
    deadline = monotonic() + TIMEOUT_SECONDS
    _remaining(deadline, abort)
    url = arguments["url"]
    if not isinstance(url, str) or not url.strip() or "\x00" in url:
        raise ToolError("invalid_arguments", "url 必须是非空公开网页地址。")
    if len(url) > MAX_URL_CHARS:
        raise ToolError("invalid_arguments", "url 过长，请提供简短地址。")
    if any(ord(character) < 32 or ord(character) == 127 for character in url):
        raise ToolError("invalid_url", "URL 不能包含控制字符。")
    try:
        parsed = urlsplit(url.strip())
    except ValueError:
        raise ToolError("invalid_url", "URL 格式无效。") from None
    if (parsed.scheme.lower() not in ALLOWED_SCHEMES
            or parsed.username is not None or parsed.password is not None):
        raise ToolError("invalid_url", "只允许不含认证信息的公开 http/https URL。")
    try:
        hostname, port, addresses = _validated_host_port(parsed, deadline, abort)
        with ExitStack() as cleanup:
            if opener is None:
                path = parsed.path or "/"
                if parsed.query:
                    path += "?" + parsed.query
                response, transport_socket = cleanup.enter_context(_open_pinned(
                    hostname, port, addresses, path,
                    secure=parsed.scheme.lower() == "https", deadline=deadline, abort=abort,
                ))
            else:
                response = cleanup.enter_context(opener(url, timeout=_remaining(deadline, abort)))
                transport_socket = None
            _remaining(deadline, abort)
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode() if hasattr(response, "getcode") else 200
            if type(status) is int and 300 <= status < 400:
                raise ToolError("redirect_not_supported", "网页发生重定向，已拒绝跟踪。")
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
            data = _read_bounded(response, transport_socket, MAX_BYTES, deadline, abort)
            if len(data) > MAX_BYTES:
                raise ToolError("too_large", f"网页超过 {MAX_BYTES} 字节，已停止读取。")
            charset = getattr(headers, "get_content_charset", lambda: None)()
            text = data.decode(charset or "utf-8", errors="replace")
            final_url = response.geturl() if opener is not None and hasattr(response, "geturl") else url
    except ToolError:
        raise
    except (HTTPError, URLError, HTTPException, TimeoutError, OSError):
        raise ToolError("fetch_error", "无法访问该网页，请检查 URL 或稍后重试。") from None

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


def _remaining(deadline, abort):
    if abort is not None and abort.is_set():
        raise ToolError("execution_cancelled", "网页读取已取消。")
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("网页读取超时。")
    return remaining


def _public_address(address):
    if not address.is_global or address.is_multicast or address.is_reserved:
        return False
    if isinstance(address, ipaddress.IPv6Address):
        return not (
            address.is_site_local or address.ipv4_mapped is not None
            or address.sixtofour is not None or address.teredo is not None
            or "%" in str(address)
            or any(address in network for network in IPV6_TRANSLATION_NETWORKS)
        )
    return True


def _dns_name(name):
    if not isinstance(name, str):
        raise ValueError("域名格式无效")
    name = name.rstrip(".").lower()
    if len(name) > 253 or not all(
        re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
        for label in name.split(".")
    ):
        raise ValueError("域名格式无效")
    return name


def _validated_host_port(parsed, deadline, abort):
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
    try:
        port = parsed.port
    except ValueError:
        raise ToolError("invalid_url", "URL 端口无效。") from None
    if port is None:
        port = 443 if parsed.scheme.lower() == "https" else 80
    if port not in ALLOWED_PORTS:
        raise ToolError("unsafe_url", "只允许使用 80 和 443 端口。")
    try:
        literal = ipaddress.ip_address(encoded)
    except ValueError:
        try:
            encoded = _dns_name(encoded)
        except ValueError:
            raise ToolError("invalid_url", "URL 主机名无效。") from None
        try:
            resolved = socket.getaddrinfo(encoded, port, type=socket.SOCK_STREAM)
        except socket.gaierror:
            raise ToolError("dns_error", "无法解析网页域名。") from None
        try:
            addresses = list(dict.fromkeys(ipaddress.ip_address(item[4][0]) for item in resolved))
        except (ValueError, IndexError, TypeError):
            raise ToolError("unsafe_url", "域名解析结果无效，已拒绝访问。") from None
        if not addresses:
            raise ToolError("dns_error", "域名没有有效地址，已拒绝访问。")
        fake = [address for address in addresses if address.version == 4 and address in FAKE_IP_NETWORK]
        if any(not _public_address(address) and address not in fake for address in addresses):
            raise ToolError("unsafe_url", "目标地址不是原生公网地址，已拒绝访问。")
        if fake:
            addresses = _resolve_doh(encoded, deadline, abort)
    else:
        if not _public_address(literal):
            raise ToolError("unsafe_url", "仅允许原生公网 IP；私网、fake-IP 和转换地址已拒绝。")
        addresses = [literal]
        encoded = str(literal)
    _remaining(deadline, abort)
    return encoded, port, addresses


@contextmanager
def _open_pinned(hostname, port, addresses, path, *, secure, deadline, abort):
    connection = HTTPConnection(hostname, port, timeout=_remaining(deadline, abort))
    response = None
    transport_socket = None
    try:
        for index, address in enumerate(addresses):
            address = ipaddress.ip_address(address)
            transport_socket = socket.socket(
                socket.AF_INET6 if address.version == 6 else socket.AF_INET, socket.SOCK_STREAM,
            )
            connection.sock = transport_socket
            try:
                transport_socket.settimeout(_remaining(deadline, abort))
                transport_socket.connect((str(address), port))
                if secure:
                    transport_socket.settimeout(_remaining(deadline, abort))
                    transport_socket = ssl.create_default_context().wrap_socket(
                        transport_socket, server_hostname=hostname,
                    )
                    connection.sock = transport_socket
                break
            except OSError:
                connection.close()
                if index == len(addresses) - 1:
                    raise
        transport_socket.settimeout(_remaining(deadline, abort))
        host = f"[{hostname}]" if ":" in hostname else hostname
        if port != (443 if secure else 80):
            host += f":{port}"
        connection.request("GET", path, headers={"Host": host, "Connection": "close"})
        transport_socket.settimeout(_remaining(deadline, abort))
        response = connection.getresponse()
        _remaining(deadline, abort)
        yield response, transport_socket
    finally:
        if response is not None:
            response.close()
        connection.close()


def _read_bounded(response, transport_socket, limit, deadline, abort):
    chunks = []
    received = 0
    read = getattr(response, "read1", response.read)
    while received <= limit:
        timeout = _remaining(deadline, abort)
        if transport_socket is not None:
            transport_socket.settimeout(timeout)
        chunk = read(min(65536, limit + 1 - received))
        _remaining(deadline, abort)
        if not chunk:
            break
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def _resolve_doh(hostname, deadline, abort):
    addresses = []
    try:
        for record_type in (1, 28):
            path = "/resolve?" + urlencode({
                "name": hostname, "type": record_type,
                "edns_client_subnet": "0.0.0.0/0",
            })
            with _open_pinned(
                "dns.google", 443, ("8.8.8.8",), path,
                secure=True, deadline=deadline, abort=abort,
            ) as (response, transport_socket):
                if response.status != 200:
                    raise ValueError("DoH HTTP 状态无效")
                data = _read_bounded(response, transport_socket, MAX_DNS_BYTES, deadline, abort)
                if len(data) > MAX_DNS_BYTES:
                    raise ValueError("DoH 响应过大")
                addresses.extend(_doh_addresses(json.loads(data), hostname, record_type))
        if not addresses:
            raise ValueError("DoH 没有有效公网地址")
    except (ValueError, UnicodeError, HTTPException, OSError):
        raise ToolError("dns_error", "本机返回 fake-IP，公开 DNS 核验失败，已拒绝访问。") from None
    return list(dict.fromkeys(addresses))


def _doh_addresses(payload, hostname, record_type):
    if not isinstance(payload, dict) or type(payload.get("Status")) is not int:
        raise ValueError("DoH 状态格式无效")
    if payload["Status"] != 0 or payload.get("TC") is not False:
        raise ValueError("DoH 解析失败或结果不完整")
    questions = payload.get("Question")
    if not isinstance(questions, list) or len(questions) != 1:
        raise ValueError("DoH 问题格式无效")
    question = questions[0]
    if (not isinstance(question, dict) or type(question.get("type")) is not int
            or question["type"] != record_type or _dns_name(question.get("name")) != hostname):
        raise ValueError("DoH 问题与请求不符")
    answers = payload.get("Answer", [])
    if not isinstance(answers, list):
        raise ValueError("DoH 答案格式无效")
    aliases = {}
    records = []
    for answer in answers:
        if not isinstance(answer, dict) or type(answer.get("type")) is not int:
            raise ValueError("DoH 记录格式无效")
        owner = _dns_name(answer.get("name"))
        kind, value = answer["type"], answer.get("data")
        if not isinstance(value, str):
            raise ValueError("DoH 记录值无效")
        if kind == 5:
            target = _dns_name(value)
            if owner in aliases and aliases[owner] != target:
                raise ValueError("DoH 别名冲突")
            aliases[owner] = target
        elif kind == record_type:
            address = ipaddress.ip_address(value)
            if address.version != (4 if record_type == 1 else 6):
                raise ValueError("DoH 地址类型不符")
            if not _public_address(address):
                raise ToolError("unsafe_url", "公开 DNS 答案含非原生公网地址，已拒绝访问。")
            records.append((owner, address))
        else:
            raise ValueError("DoH 返回了未请求的记录类型")
    chain = {hostname}
    target = hostname
    while target in aliases:
        target = aliases[target]
        if target in chain:
            raise ValueError("DoH 别名循环")
        chain.add(target)
    if any(owner not in chain for owner in aliases) or any(owner != target for owner, _ in records):
        raise ValueError("DoH 答案与请求域名无关")
    return [address for _, address in records]


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
