from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import ssl
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

from harness.permissions import PermissionPolicy
from harness.tools import REGISTRY, create_tool_executor, get_tool_definitions
from harness.tools.executor import ToolError
from harness.tools.web_fetch import DEFINITION, execute


class FakeResponse(io.BytesIO):
    def __init__(self, data, *, status=200, content_type="text/html", url="https://example.com"):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Type": content_type, "Content-Length": str(len(data))}
        self.final_url = url

    def geturl(self):
        return self.final_url


def wire_response(body, *, status=200, content_type="text/html"):
    if isinstance(body, dict):
        body = json.dumps(body).encode()
        content_type = "application/json"
    return (
        f"HTTP/1.1 {status} Response\r\nContent-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
        + body
    )


def dns_response(hostname, record_type, answers=()):
    return {
        "Status": 0, "TC": False,
        "Question": [{"name": hostname + ".", "type": record_type}],
        "Answer": [
            {"name": owner + ".", "type": kind, "TTL": 300, "data": value}
            for owner, kind, value in answers
        ],
    }


class WireSocket:
    def __init__(self, response):
        self.response = response
        self.connected = None
        self.sent = b""
        self.closed = False
        self.reader = None
        self.timeouts = []

    def settimeout(self, timeout):
        self.timeouts.append(timeout)

    def connect(self, address):
        self.connected = address

    def sendall(self, data):
        self.sent += data

    def makefile(self, mode):
        self.reader = io.BytesIO(self.response)
        return self.reader

    def close(self):
        self.closed = True


class WebFetchTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def network(self, *responses, addresses=("93.184.216.34",)):
        sockets = [WireSocket(response) for response in responses]
        factories = {
            "socket.getaddrinfo": Mock(return_value=[
                (2 if ":" not in address else 10, 1, 6, "", (address, 443))
                for address in addresses
            ]),
            "socket.socket": Mock(side_effect=sockets),
            "ssl.create_default_context": Mock(),
        }
        context = factories["ssl.create_default_context"].return_value
        context.wrap_socket.side_effect = lambda sock, **kwargs: sock
        for name, mock in factories.items():
            patcher = patch(name, mock)
            patcher.start()
            self.addCleanup(patcher.stop)
        return sockets, factories, context

    def test_default_transport_pins_ip_and_preserves_host_and_tls_name(self):
        sockets, factories, context = self.network(wire_response(b"<p>public page</p>"))
        result = execute({"url": "https://example.com/report?q=1#fragment"}, self.root)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["final_url"], "https://example.com/report?q=1#fragment")
        self.assertEqual(sockets[0].connected[0], "93.184.216.34")
        self.assertIn(b"GET /report?q=1 HTTP/1.1\r\n", sockets[0].sent)
        self.assertIn(b"Host: example.com\r\n", sockets[0].sent)
        self.assertNotIn(b"fragment", sockets[0].sent)
        self.assertEqual(context.wrap_socket.call_args.kwargs["server_hostname"], "example.com")
        factories["socket.getaddrinfo"].assert_called_once()
        self.assertTrue(sockets[0].closed)
        self.assertTrue(sockets[0].reader.closed)

    def test_fake_ip_uses_pinned_doh_before_pinned_web_connection(self):
        sockets, factories, context = self.network(
            wire_response(dns_response("example.com", 1, [("example.com", 1, "93.184.216.34")])),
            wire_response(dns_response("example.com", 28)),
            wire_response(b"<p>verified page</p>"),
            addresses=("198.18.109.237",),
        )
        result = execute({"url": "https://example.com/private-path?credential=not-for-dns"}, self.root)
        self.assertIn("verified page", result["text"])
        self.assertEqual([sock.connected[0] for sock in sockets], ["8.8.8.8", "8.8.8.8", "93.184.216.34"])
        self.assertEqual(
            [call.kwargs["server_hostname"] for call in context.wrap_socket.call_args_list],
            ["dns.google", "dns.google", "example.com"],
        )
        for sock, record_type in zip(sockets[:2], ("1", "28")):
            target = sock.sent.decode().splitlines()[0].split()[1]
            self.assertEqual(urlsplit(target).path, "/resolve")
            self.assertEqual(parse_qs(urlsplit(target).query), {
                "name": ["example.com"], "type": [record_type],
                "edns_client_subnet": ["0.0.0.0/0"],
            })
            self.assertIn(b"Host: dns.google\r\n", sock.sent)
            self.assertNotIn(b"private-path", sock.sent)
            self.assertNotIn(b"credential", sock.sent)
        factories["socket.getaddrinfo"].assert_called_once()
        self.assertTrue(all(sock.closed and sock.reader.closed for sock in sockets))

    def test_public_ipv4_and_native_ipv6_literals_use_default_transport(self):
        sockets, factories, _context = self.network(
            wire_response(b"IPv4"), wire_response(b"IPv6"),
        )
        for url in ("http://8.8.8.8/", "http://[2606:4700:4700::1111]/"):
            with self.subTest(url=url):
                self.assertEqual(execute({"url": url}, self.root)["status"], "success")
        self.assertEqual(sockets[0].connected[0], "8.8.8.8")
        self.assertEqual(sockets[1].connected[0], "2606:4700:4700::1111")
        self.assertIn(b"Host: [2606:4700:4700::1111]\r\n", sockets[1].sent)
        factories["socket.getaddrinfo"].assert_not_called()

    def test_shared_multicast_and_transition_ipv6_addresses_stay_blocked(self):
        opener = Mock()
        for address in (
            "198.18.1.1", "198.19.255.255", "100.64.0.1", "224.0.0.1", "fec0::1", "ff02::1",
            "::ffff:127.0.0.1", "::ffff:8.8.8.8", "2002:7f00:1::1",
            "64:ff9b::c0a8:101", "2001:0000:4136:e378:8000:63bf:3fff:fdd2",
        ):
            with self.subTest(address=address):
                host = f"[{address}]" if ":" in address else address
                with self.assertRaises(ToolError) as error:
                    execute({"url": f"http://{host}/"}, self.root, opener=opener)
                self.assertEqual(error.exception.code, "unsafe_url")
        opener.assert_not_called()

    def test_fake_ip_mixed_with_private_or_shared_address_never_uses_doh(self):
        sockets, factories, _context = self.network()
        for address in ("192.168.1.1", "100.64.0.1", "224.0.0.1", "fec0::1"):
            factories["socket.getaddrinfo"].return_value = [
                (2, 1, 6, "", ("198.18.1.1", 443)),
                (2, 1, 6, "", (address, 443)),
            ]
            with self.subTest(address=address):
                with self.assertRaises(ToolError) as error:
                    execute({"url": "https://example.com/"}, self.root)
                self.assertEqual(error.exception.code, "unsafe_url")
        factories["socket.socket"].assert_not_called()

    def test_cancel_before_resolving_never_calls_dns_or_transport(self):
        abort = Event()
        abort.set()
        _sockets, factories, _context = self.network()
        with self.assertRaises(ToolError) as error:
            execute({"url": "https://example.com/"}, self.root, abort=abort)
        self.assertEqual(error.exception.code, "execution_cancelled")
        factories["socket.getaddrinfo"].assert_not_called()
        factories["socket.socket"].assert_not_called()

    def test_valid_doh_cname_chain_connects_only_terminal_public_address(self):
        sockets, factories, _context = self.network(
            wire_response(dns_response("example.com", 1, [
                ("example.com", 5, "alias.example.com."),
                ("alias.example.com", 5, "edge.example.com."),
                ("edge.example.com", 1, "93.184.216.34"),
            ])),
            wire_response(dns_response("example.com", 28)),
            wire_response(b"CNAME page"),
            addresses=("198.19.1.1",),
        )
        self.assertEqual(execute({"url": "https://example.com"}, self.root)["text"], "CNAME page")
        self.assertEqual(sockets[-1].connected[0], "93.184.216.34")
        factories["socket.getaddrinfo"].assert_called_once()

    def test_invalid_doh_status_questions_answers_and_aliases_fail_closed(self):
        invalid = []
        for field, value in (("Status", True), ("Status", 2), ("TC", True), ("TC", None),
                             ("Question", []), ("Question", [{"name": "other.example.", "type": 1}]),
                             ("Question", [{"name": "example.com.", "type": 28}]),
                             ("Answer", {})):
            payload = dns_response("example.com", 1)
            payload[field] = value
            invalid.append(payload)
        for answers in (
            [("other.example", 1, "93.184.216.34")],
            [("example.com", 28, "2606:4700:4700::1111")],
            [("example.com", 1, "2606:4700:4700::1111")],
            [("example.com", 5, "alias.example."), ("example.com", 5, "other.example.")],
            [("example.com", 5, "alias.example."), ("alias.example", 5, "example.com.")],
            [("example.com", 5, "alias.example."), ("example.com", 1, "93.184.216.34")],
            [("unrelated.example", 5, "alias.example.")],
        ):
            invalid.append(dns_response("example.com", 1, answers))
        for payload in invalid:
            with self.subTest(payload=payload):
                sockets, factories, _context = self.network(
                    wire_response(payload), addresses=("198.18.1.1",),
                )
                with self.assertRaises(ToolError) as error:
                    execute({"url": "https://example.com"}, self.root)
                self.assertEqual(error.exception.code, "dns_error")
                self.assertEqual(factories["socket.socket"].call_count, 1)
                self.assertTrue(sockets[0].closed and sockets[0].reader.closed)

    def test_doh_non_public_answers_fail_closed_without_web_connection(self):
        for address in ("127.0.0.1", "192.168.1.1", "100.64.0.1", "224.0.0.1", "198.18.1.1"):
            with self.subTest(address=address):
                sockets, factories, _context = self.network(
                    wire_response(dns_response("example.com", 1, [("example.com", 1, address)])),
                    addresses=("198.18.1.1",),
                )
                with self.assertRaises(ToolError) as error:
                    execute({"url": "https://example.com"}, self.root)
                self.assertEqual(error.exception.code, "unsafe_url")
                self.assertEqual(factories["socket.socket"].call_count, 1)
                self.assertTrue(sockets[0].closed and sockets[0].reader.closed)

    def test_doh_ipv6_answers_accept_native_public_and_reject_private_or_translated(self):
        for address in ("2606:4700:4700::1111", "fec0::1", "ff02::1", "::ffff:127.0.0.1",
                        "2002:7f00:1::1", "64:ff9b::c0a8:101"):
            with self.subTest(address=address):
                sockets, factories, _context = self.network(
                    wire_response(dns_response("example.com", 1)),
                    wire_response(dns_response("example.com", 28, [("example.com", 28, address)])),
                    wire_response(b"IPv6 page"), addresses=("198.18.1.1",),
                )
                if address == "2606:4700:4700::1111":
                    self.assertEqual(execute({"url": "https://example.com"}, self.root)["text"], "IPv6 page")
                    self.assertEqual(sockets[-1].connected[0], address)
                    self.assertTrue(sockets[-1].closed)
                else:
                    with self.assertRaises(ToolError) as error:
                        execute({"url": "https://example.com"}, self.root)
                    self.assertEqual(error.exception.code, "unsafe_url")
                    self.assertEqual(factories["socket.socket"].call_count, 2)
                self.assertTrue(all(sock.closed and sock.reader.closed for sock in sockets[:2]))

    def test_doh_invalid_json_oversize_http_error_and_empty_answers_fail_closed(self):
        for responses in (
            (wire_response(b"{bad json", content_type="application/json"),),
            (wire_response(b" " * 65537, content_type="application/json"),),
            (wire_response(b"", status=302),),
            (wire_response(dns_response("example.com", 1)),
             wire_response(dns_response("example.com", 28))),
        ):
            with self.subTest(responses=len(responses)):
                sockets, factories, _context = self.network(*responses, addresses=("198.18.1.1",))
                with self.assertRaises(ToolError) as error:
                    execute({"url": "https://example.com"}, self.root)
                self.assertEqual(error.exception.code, "dns_error")
                self.assertEqual(factories["socket.socket"].call_count, len(responses))
                self.assertTrue(all(sock.closed and sock.reader.closed for sock in sockets))

    def test_default_transport_redirect_is_rejected_and_closed(self):
        sockets, factories, _context = self.network(wire_response(b"redirect", status=302))
        with self.assertRaises(ToolError) as error:
            execute({"url": "https://example.com"}, self.root)
        self.assertEqual(error.exception.code, "redirect_not_supported")
        self.assertEqual(factories["socket.socket"].call_count, 1)
        self.assertTrue(sockets[0].closed and sockets[0].reader.closed)

    def test_default_tls_context_requires_certificate_and_hostname_verification(self):
        secure_context = ssl.create_default_context()
        sockets, factories, _context = self.network(wire_response(b"TLS page"))
        factories["ssl.create_default_context"].return_value = secure_context
        with patch.object(secure_context, "wrap_socket", side_effect=lambda sock, **kwargs: sock):
            execute({"url": "https://example.com"}, self.root)
        self.assertTrue(secure_context.check_hostname)
        self.assertEqual(secure_context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(sockets[0].closed)

    def test_tls_failure_closes_socket_without_sending_request(self):
        sockets, _factories, context = self.network(wire_response(b"unused"))
        context.wrap_socket.side_effect = ssl.SSLCertVerificationError("invalid certificate")
        with self.assertRaises(ToolError) as error:
            execute({"url": "https://example.com"}, self.root)
        self.assertEqual(error.exception.code, "fetch_error")
        self.assertTrue(sockets[0].closed)
        self.assertEqual(sockets[0].sent, b"")

    def test_failed_public_address_connection_retries_only_other_validated_address(self):
        sockets, factories, _context = self.network(
            wire_response(b"unused"), wire_response(b"second address"),
            addresses=("93.184.216.34", "8.8.4.4"),
        )
        sockets[0].connect = Mock(side_effect=OSError("connection failed"))
        self.assertEqual(execute({"url": "http://example.com"}, self.root)["text"], "second address")
        self.assertTrue(all(sock.closed for sock in sockets))
        self.assertEqual(sockets[1].connected[0], "8.8.4.4")
        factories["socket.getaddrinfo"].assert_called_once()

    def test_cancellation_after_dns_prevents_connection(self):
        abort = Event()
        executor = create_tool_executor(self.root, abort=abort)
        _sockets, factories, _context = self.network()
        addresses = factories["socket.getaddrinfo"].return_value

        def resolve(*args, **kwargs):
            abort.set()
            return addresses

        factories["socket.getaddrinfo"].side_effect = resolve
        result = executor("web_fetch", {"url": "https://example.com"})
        self.assertEqual(result["code"], "execution_cancelled")
        factories["socket.socket"].assert_not_called()

    def test_cancellation_during_body_read_closes_response_and_socket(self):
        abort = Event()
        sockets, _factories, _context = self.network(wire_response(b"partial body"))

        class CancellingReader(io.BytesIO):
            def read1(self, size=-1):
                data = super().read1(size)
                abort.set()
                return data

        sockets[0].reader = CancellingReader(sockets[0].response)
        sockets[0].makefile = Mock(return_value=sockets[0].reader)
        with self.assertRaises(ToolError) as error:
            execute({"url": "https://example.com"}, self.root, abort=abort)
        self.assertEqual(error.exception.code, "execution_cancelled")
        self.assertTrue(sockets[0].closed and sockets[0].reader.closed)

    def test_doh_and_web_connection_share_total_deadline(self):
        sockets, _factories, _context = self.network(
            wire_response(dns_response("example.com", 1, [("example.com", 1, "93.184.216.34")])),
            wire_response(dns_response("example.com", 28)), wire_response(b"unused"),
            addresses=("198.18.1.1",),
        )
        elapsed = [0]

        def connect(address):
            elapsed[0] += 4

        for sock in sockets:
            sock.connect = Mock(side_effect=connect)
        with patch("harness.tools.web_fetch.monotonic", side_effect=lambda: elapsed[0]):
            with self.assertRaises(ToolError) as error:
                execute({"url": "https://example.com"}, self.root)
        self.assertEqual(error.exception.code, "fetch_error")
        self.assertEqual([sock.timeouts[0] for sock in sockets], [10, 6, 2])
        self.assertTrue(all(sock.closed for sock in sockets))
        self.assertTrue(all(sock.reader.closed for sock in sockets[:2]))
        self.assertIsNone(sockets[-1].reader)

    def test_explicit_deny_prevents_dns_transport_and_confirmation(self):
        confirm = Mock(return_value=True)
        executor = create_tool_executor(
            self.root, permissions=PermissionPolicy(deny=["web_fetch"]), confirm=confirm,
        )
        _sockets, factories, _context = self.network()
        result = executor("web_fetch", {"url": "https://example.com"})
        self.assertEqual(result["code"], "permission_denied")
        factories["socket.getaddrinfo"].assert_not_called()
        factories["socket.socket"].assert_not_called()
        confirm.assert_not_called()

    def test_concurrent_requests_keep_dns_host_and_socket_isolated(self):
        addresses = {"one.example": "8.8.8.8", "two.example": "8.8.4.4"}
        sockets = []

        class RoutedSocket(WireSocket):
            def connect(self, address):
                super().connect(address)
                self.response = wire_response(address[0].encode())

        def resolve(hostname, port, **kwargs):
            return [(2, 1, 6, "", (addresses[hostname], port))]

        def socket_factory(*args):
            sock = RoutedSocket(b"")
            sockets.append(sock)
            return sock

        with patch("harness.tools.web_fetch.socket.getaddrinfo", side_effect=resolve), \
                patch("harness.tools.web_fetch.socket.socket", side_effect=socket_factory):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(execute, {"url": f"http://{host}/"}, self.root)
                           for host in addresses]
                self.assertEqual([future.result()["text"] for future in futures], list(addresses.values()))
        for sock in sockets:
            hostname = next(host for host, address in addresses.items() if address == sock.connected[0])
            self.assertIn(f"Host: {hostname}\r\n".encode(), sock.sent)
            self.assertTrue(sock.closed and sock.reader.closed)

    def fetch(self, url, response, *, opener=None):
        patcher = patch("harness.tools.web_fetch.socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("93.184.216.34", 443)),
        ])
        patcher.start()
        self.addCleanup(patcher.stop)
        return execute({"url": url}, self.root, opener=opener or Mock(return_value=response))

    def test_definition_is_registered_and_reads_public_html(self):
        names = [item["function"]["name"] for item in get_tool_definitions()]
        self.assertIn("web_fetch", names)
        self.assertIs(REGISTRY.get("web_fetch")[0], DEFINITION)
        self.assertTrue(DEFINITION.supports_cancellation)
        response = FakeResponse(
            "<html><head><title>产品页</title><script>secret()</script></head>"
            "<body><h1>Personal Agent</h1><p>竞品信息。</p></body></html>".encode(),
        )
        result = self.fetch("https://example.com", response)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["title"], "产品页")
        self.assertIn("Personal Agent", result["text"])
        self.assertNotIn("secret()", result["text"])

    def test_private_loopback_non_standard_ports_and_bad_schemes_are_rejected(self):
        for url in (
            "http://127.0.0.1/", "http://localhost/", "http://192.168.1.1/",
            "http://example.local/", "ftp://example.com/file", "https://user:pass@example.com/",
            "https://example.com:8443/", "https://@example.com/", "https://example.com/\n",
            "https://example.com/\x7f", "https://example.com/\t",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ToolError):
                    execute({"url": url}, self.root)

    def test_http_status_content_type_and_size_limits_are_errors(self):
        with self.assertRaises(ToolError) as error:
            self.fetch("https://example.com", FakeResponse(b"", status=404))
        self.assertEqual(error.exception.code, "http_error")
        with self.assertRaises(ToolError) as error:
            self.fetch("https://example.com", FakeResponse(b"x", content_type="image/png"))
        self.assertEqual(error.exception.code, "unsupported_content")
        with self.assertRaises(ToolError) as error:
            self.fetch("https://example.com", FakeResponse(b"x" * (1_048_576 + 1)))
        self.assertIn(error.exception.code, {"too_large"})

    def test_default_permission_allows_repeated_reads_without_confirmation(self):
        confirm = Mock(return_value=False)
        executor = create_tool_executor(self.root, confirm=confirm)
        sockets, factories, _context = self.network(*[
            wire_response("<html><body>公开内容</body></html>".encode()) for _ in range(3)
        ])
        for path in ("/research", "/news", "/projects"):
            with self.subTest(path=path):
                result = executor("web_fetch", {"url": "https://example.com" + path})
                self.assertEqual(result["status"], "success")
                self.assertTrue(result["executed"])
                self.assertIn("公开内容", result["text"])
        self.assertEqual(factories["socket.socket"].call_count, 3)
        self.assertTrue(all(sock.closed for sock in sockets))
        confirm.assert_not_called()

    def test_default_permission_keeps_private_hosts_blocked(self):
        confirm = Mock(return_value=True)
        executor = create_tool_executor(self.root, confirm=confirm)
        _sockets, factories, _context = self.network(addresses=("192.168.1.1",))
        for url in ("http://localhost", "https://internal.example.com"):
            with self.subTest(url=url):
                result = executor("web_fetch", {"url": url})
                self.assertEqual(result["code"], "unsafe_url")
        factories["socket.socket"].assert_not_called()
        confirm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
