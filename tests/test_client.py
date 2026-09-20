import io
import json
import unittest
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError

from harness.client import APIError, ChatCompletionClient, DeepSeekClient


def event(chunk):
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")


def chunk(delta=None, *, finish_reason=None, **metadata):
    return {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}], **metadata}


def stream(*chunks, done=True):
    return b"".join(event(item) for item in chunks) + (b"data: [DONE]\n\n" if done else b"")


class ClientTests(unittest.TestCase):
    def test_api_error_defaults_and_usage_context(self):
        usage = {"prompt_tokens": 2, "completion_tokens": 1}
        error = APIError("请求失败", response={
            "model": "deepseek-flash", "created": 123, "usage": usage,
            "choices": [{"message": {"content": "private-response"}}],
        })
        self.assertIsNone(error.status_code)
        self.assertFalse(error.retryable)
        self.assertTrue(error.request_attempted)
        self.assertEqual(error.response, {"model": "deepseek-flash", "created": 123, "usage": usage})
        self.assertNotIn("private-response", str(error))

    def test_official_request_contract(self):
        messages = [{"role": "user", "content": "你好"}]
        tools = [{"type": "function", "function": {"name": "read_file"}}]
        usage = {"prompt_tokens": 2, "completion_tokens": 1, "prompt_cache_hit_tokens": 0}

        def opener(request, *, timeout):
            self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
            self.assertEqual(request.get_header("Content-type"), "application/json")
            self.assertEqual(timeout, 120)
            self.assertEqual(json.loads(request.data), {
                "model": "deepseek-flash", "messages": messages, "tools": tools,
                "tool_choice": "auto", "thinking": {"type": "disabled"}, "stream": True,
                "stream_options": {"include_usage": True},
            })
            return io.BytesIO(stream(
                chunk({"role": "assistant", "content": "你好"}, model="deepseek-flash", created=123),
                chunk(finish_reason="stop", usage=usage),
            ))

        client = DeepSeekClient("test-key", opener=opener)
        self.assertEqual(client.complete(model="deepseek-flash", messages=messages, tools=tools), {
            "model": "deepseek-flash", "created": 123, "usage": usage,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "你好",
            }}],
        })

    def test_glm_request_contract_and_usage_cache_normalization(self):
        messages = [{"role": "user", "content": "你好"}]
        usage = {
            "prompt_tokens": 8,
            "completion_tokens": 2,
            "total_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 3},
        }

        def opener(request, *, timeout):
            self.assertEqual(request.full_url, "https://open.bigmodel.cn/api/paas/v4/chat/completions")
            self.assertEqual(request.get_header("Authorization"), "Bearer glm-test-key")
            body = json.loads(request.data)
            self.assertEqual(body["model"], "glm-5.3-flash")
            self.assertEqual(body["thinking"], {"type": "enabled"})
            self.assertEqual(body["reasoning_effort"], "low")
            self.assertEqual(timeout, 120)
            return io.BytesIO(stream(
                chunk({"content": "你好"}, model="glm-5.3-flash", created=123),
                chunk(finish_reason="stop", usage=usage),
            ))

        client = ChatCompletionClient(
            "glm-test-key", opener=opener, provider="glm",
        )
        response = client.complete(
            model=client.model, messages=messages, tools=[],
        )
        self.assertEqual(response["usage"]["prompt_cache_hit_tokens"], 3)
        self.assertEqual(response["usage"]["prompt_cache_miss_tokens"], 5)

    def test_glm_api_key_error_names_correct_provider(self):
        with self.assertRaisesRegex(ValueError, "GLM_API_KEY"):
            ChatCompletionClient("", provider="glm")

    def test_summary_request_disables_tools_and_limits_output(self):
        requests = []

        def opener(request, **kwargs):
            requests.append(json.loads(request.data))
            return io.BytesIO(stream(chunk({"content": "摘要"}, finish_reason="stop")))

        client = DeepSeekClient("test-key", opener=opener)
        response = client.complete(
            model="deepseek-flash", messages=[{"role": "user", "content": "总结旧对话"}],
            tools=[], max_tokens=1000,
        )
        self.assertEqual(response["choices"][0]["message"]["content"], "摘要")
        self.assertEqual(requests[0]["tools"], [])
        self.assertEqual(requests[0]["tool_choice"], "none")
        self.assertEqual(requests[0]["max_tokens"], 1000)

        client.complete(model="deepseek-flash", messages=[], tools=[], max_tokens=None)
        self.assertNotIn("max_tokens", requests[1])

    def test_text_callback_runs_before_reading_next_chunk(self):
        seen = []
        prefix = event(chunk({"content": "你"}))
        testcase = self

        class ObservedResponse(io.BytesIO):
            def __next__(self):
                if self.tell() == len(prefix):
                    testcase.assertEqual(seen, ["你"])
                return super().__next__()

        body = prefix + stream(chunk({"content": "好，世界"}, finish_reason="stop"))
        client = DeepSeekClient("test-key", opener=lambda *args, **kwargs: ObservedResponse(body))
        response = client.complete(model="deepseek-flash", messages=[], tools=[], on_text=seen.append)
        self.assertEqual(seen, ["你", "好，世界"])
        self.assertEqual(response["choices"][0]["message"]["content"], "你好，世界")

    def test_tool_calls_are_assembled_by_index(self):
        body = stream(
            chunk({"tool_calls": [
                {"index": 1, "id": "call_", "type": "function", "function": {
                    "name": "run_", "arguments": '{"command":',
                }},
                {"index": 0, "id": "call_0", "type": "function", "function": {
                    "name": "read_", "arguments": '{"path":"',
                }},
            ]}),
            chunk({"tool_calls": [
                {"index": 0, "function": {"name": "file", "arguments": '中文.txt"}'}},
                {"index": 1, "id": "1", "function": {"name": "command", "arguments": '"pwd"}'}},
            ]}, finish_reason="tool_calls"),
        )
        seen = []
        client = DeepSeekClient("test-key", opener=lambda *args, **kwargs: io.BytesIO(body))
        response = client.complete(model="deepseek-flash", messages=[], tools=[], on_text=seen.append)
        self.assertEqual(response["choices"][0]["message"]["tool_calls"], [
            {"id": "call_0", "type": "function", "function": {
                "name": "read_file", "arguments": '{"path":"中文.txt"}',
            }},
            {"id": "call_1", "type": "function", "function": {
                "name": "run_command", "arguments": '{"command":"pwd"}',
            }},
        ])
        self.assertEqual(seen, [])
        self.assertEqual(response["choices"][0]["finish_reason"], "tool_calls")

    def test_keepalive_and_usage_only_chunk(self):
        usage = {"prompt_tokens": 20, "completion_tokens": 5,
                 "prompt_cache_hit_tokens": 12, "prompt_cache_miss_tokens": 8}
        body = b": keepalive\r\n\r\n" + event(chunk({"content": "回答"}, finish_reason="stop"))
        body += event({"choices": [], "usage": usage}) + b"data: [DONE]"
        client = DeepSeekClient("test-key", opener=lambda *args, **kwargs: io.BytesIO(body))
        response = client.complete(model="deepseek-flash", messages=[], tools=[])
        self.assertEqual(response["usage"], usage)
        self.assertEqual(response["choices"][0]["message"]["content"], "回答")

    def test_incomplete_stream_requires_done_and_finish_reason(self):
        usage = {"prompt_tokens": 2, "completion_tokens": 1}
        for body, expected_usage in [
            (stream(chunk({"content": "未完成"}), done=False), None),
            (stream(chunk({"content": "未完成"})), None),
            (stream(chunk({"content": "回答"}, finish_reason="stop", usage=usage,
                          model="deepseek-flash", created=123), done=False), usage),
        ]:
            client = DeepSeekClient("test-key", opener=lambda *args, **kwargs: io.BytesIO(body))
            with self.subTest(body=body), self.assertRaisesRegex(APIError, "未完成") as result:
                client.complete(model="deepseek-flash", messages=[], tools=[])
            self.assertEqual(getattr(result.exception, "response", {}).get("usage"), expected_usage)
            self.assertTrue(result.exception.retryable)
            self.assertIsNone(result.exception.status_code)
            if expected_usage:
                self.assertEqual(result.exception.response["model"], "deepseek-flash")
                self.assertEqual(result.exception.response["created"], 123)

    def test_stream_preserves_length_finish_reason_for_engine(self):
        body = stream(chunk({"content": "截断"}, finish_reason="length"))
        client = DeepSeekClient("test-key", opener=lambda *args, **kwargs: io.BytesIO(body))
        response = client.complete(model="deepseek-flash", messages=[], tools=[])
        self.assertEqual(response["choices"][0]["finish_reason"], "length")

    def test_missing_key(self):
        for key in [None, "", "   "]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                DeepSeekClient(key)

    def test_http_error_is_readable_and_does_not_leak_body(self):
        requests = []

        def opener(request, **kwargs):
            requests.append(request)
            raise HTTPError(request.full_url, 401, "sensitive-details", {}, io.BytesIO(b"private-server-body"))

        client = DeepSeekClient("test-key", opener=opener)
        with self.assertRaisesRegex(APIError, "401") as result:
            client.complete(model="deepseek-flash", messages=[], tools=[])
        self.assertTrue(result.exception.request_attempted)
        self.assertEqual(result.exception.status_code, 401)
        self.assertFalse(result.exception.retryable)
        self.assertNotIn("private-server-body", str(result.exception))
        self.assertNotIn("sensitive-details", str(result.exception))
        self.assertNotIn("test-key", str(result.exception))
        self.assertEqual(len(requests), 1)

    def test_only_transient_http_errors_are_retryable(self):
        for status in [400, 401, 402, 403, 404, 422, 429, 500, 502, 503, 504]:
            body = io.BytesIO(b"private-server-body")

            def opener(request, **kwargs):
                raise HTTPError(request.full_url, status, "private-details", {}, body)

            client = DeepSeekClient("test-key", opener=opener)
            with self.subTest(status=status), self.assertRaises(APIError) as result:
                client.complete(model="deepseek-flash", messages=[], tools=[])
            self.assertEqual(result.exception.status_code, status)
            self.assertEqual(result.exception.retryable, status in {429, 500, 502, 503, 504})
            self.assertTrue(body.closed)
            self.assertNotIn("private", str(result.exception))

    def test_network_timeout_and_invalid_json(self):
        for error in [URLError("private-details"), TimeoutError("private-details"),
                      URLError(TimeoutError("private-details")), OSError("private-details")]:
            def opener(request, **kwargs):
                raise error
            client = DeepSeekClient("test-key", opener=opener)
            with self.subTest(error=type(error).__name__), self.assertRaises(APIError) as result:
                client.complete(model="deepseek-flash", messages=[], tools=[])
            self.assertNotIn("private-details", str(result.exception))
            self.assertTrue(result.exception.retryable)
            self.assertIsNone(result.exception.status_code)
        client = DeepSeekClient("test-key", opener=lambda *args, **kwargs: io.BytesIO(b"data: not-json\n\n"))
        with self.assertRaisesRegex(APIError, "JSON") as result:
            client.complete(model="deepseek-flash", messages=[], tools=[])
        self.assertFalse(result.exception.retryable)

    def test_incomplete_http_body_is_retryable_with_unknown_usage(self):
        class BrokenResponse(io.BytesIO):
            def __next__(self):
                raise IncompleteRead(b"private-partial-response", 100)

        client = DeepSeekClient("test-key", opener=lambda *args, **kwargs: BrokenResponse())
        with self.assertRaisesRegex(APIError, "连接中断") as result:
            client.complete(model="deepseek-flash", messages=[], tools=[])
        self.assertNotIn("private-partial-response", str(result.exception))
        self.assertTrue(result.exception.request_attempted)
        self.assertTrue(result.exception.retryable)
        self.assertFalse(hasattr(result.exception, "response"))

    def test_stream_timeout_keeps_received_usage_without_leaking_details(self):
        usage = {"prompt_tokens": 2, "completion_tokens": 1}

        class BrokenResponse(io.BytesIO):
            def __next__(self):
                if self.tell() == len(self.getvalue()):
                    raise TimeoutError("private-details test-key")
                return super().__next__()

        body = stream(chunk(finish_reason="stop", usage=usage), done=False)
        client = DeepSeekClient("test-key", opener=lambda *args, **kwargs: BrokenResponse(body))
        with self.assertRaisesRegex(APIError, "超时") as result:
            client.complete(model="deepseek-flash", messages=[], tools=[])
        self.assertEqual(result.exception.response["usage"], usage)
        self.assertTrue(result.exception.retryable)
        self.assertNotIn("choices", result.exception.response)
        self.assertNotIn("private-details", str(result.exception))
        self.assertNotIn("test-key", str(result.exception))

    def test_malformed_stream_errors_are_safe(self):
        for data in [b"data: private-server-body\n\n", b"data: \xff\n\n",
                     event([]), event({"choices": "private-server-body"}),
                     event(chunk({"content": {"private": "test-key"}})),
                     event(chunk({"tool_calls": [{"index": "test-key"}]}))]:
            client = DeepSeekClient("test-key", opener=lambda *args, **kwargs: io.BytesIO(data))
            with self.subTest(data=data), self.assertRaises(APIError) as result:
                client.complete(model="deepseek-flash", messages=[], tools=[])
            self.assertTrue(result.exception.request_attempted)
            self.assertFalse(result.exception.retryable)
            self.assertNotIn("private-server-body", str(result.exception))
            self.assertNotIn("test-key", str(result.exception))


if __name__ == "__main__":
    unittest.main()
