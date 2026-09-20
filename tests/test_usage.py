from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import unittest
from unittest.mock import patch

from harness.usage import PRICING, UsageLedger, format_cost, format_usage


FIXED_PRICING = {key: value for key, value in PRICING.items() if key != "peak"}
USAGE = {
    "prompt_tokens": 1000,
    "completion_tokens": 200,
    "total_tokens": 1200,
    "prompt_cache_hit_tokens": 800,
    "prompt_cache_miss_tokens": 200,
}


def created_at(date):
    return datetime.fromisoformat(date.replace("Z", "+00:00")).timestamp()


def record_usage(ledger, **overrides):
    return ledger.record(**{"usage": USAGE, "model": "deepseek-test", "turn": 1, **overrides})


class UsageLedgerTests(unittest.TestCase):
    def test_cache_is_included_in_prompt_and_billed_separately(self):
        ledger = UsageLedger(pricing=FIXED_PRICING)
        record = record_usage(ledger)
        self.assertEqual(record["prompt_tokens"], 1000)
        self.assertEqual(record["total_tokens"], 1200)
        self.assertEqual(record["cache_hit_tokens"], 800)
        self.assertEqual(record["cache_miss_tokens"], 200)
        self.assertAlmostEqual(record["estimated_cost"], 0.0001524)
        self.assertFalse(record["estimated_cache"])
        self.assertFalse(record["usage_missing"])
        self.assertEqual(record["rate_period"], "fixed")
        self.assertIn("USD 0.00015240", format_usage(record))

    def test_tool_rounds_and_later_turns_are_all_counted(self):
        ledger = UsageLedger(pricing=FIXED_PRICING)
        record_usage(ledger)
        record_usage(ledger)
        record_usage(ledger, turn=2)
        self.assertEqual([(item["request"], item["turn"]) for item in ledger.records], [(1, 1), (2, 1), (3, 2)])
        summary = ledger.summary()
        expected = {
            "requests": 3, "prompt_tokens": 3000, "completion_tokens": 600,
            "total_tokens": 3600, "cache_hit_tokens": 2400, "cache_miss_tokens": 600,
            "missing_usage_requests": 0,
        }
        for key, value in expected.items():
            self.assertEqual(summary[key], value)
        self.assertAlmostEqual(summary["estimated_cost"], 0.0004572)
        self.assertIn("请求 #3／对话 2", format_cost(ledger))

    def test_missing_or_invalid_usage_remains_unknown(self):
        ledger = UsageLedger(pricing=FIXED_PRICING)
        invalid_usages = [
            None, {}, {**USAGE, "total_tokens": 999},
            {**USAGE, "prompt_tokens": -1}, {**USAGE, "completion_tokens": 1.5},
            {**USAGE, "prompt_tokens": "1000"}, {**USAGE, "total_tokens": 2**53},
            {"prompt_tokens": True, "completion_tokens": 0, "total_tokens": 1},
        ]
        for usage in invalid_usages:
            with self.subTest(usage=usage):
                record = record_usage(ledger, usage=usage)
                self.assertTrue(record["usage_missing"])
                for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cache_hit_tokens", "cache_miss_tokens", "estimated_cost"):
                    self.assertIsNone(record[key])
                self.assertIn("费用未知", format_usage(record))
                self.assertNotIn("0.00000000", format_usage(record))
        record_usage(ledger)
        self.assertEqual(ledger.summary()["missing_usage_requests"], 8)
        self.assertEqual(ledger.summary()["total_tokens"], 1200)
        self.assertIn("预估费用已知小计：USD 0.00015240", format_cost(ledger))
        self.assertIn("8 次请求用量缺失", format_cost(ledger))

    def test_missing_or_inconsistent_cache_is_estimated_as_all_misses(self):
        ledger = UsageLedger(pricing=FIXED_PRICING)
        no_cache = {key: value for key, value in USAGE.items() if "cache" not in key}
        inputs = [
            no_cache, {**no_cache, "prompt_cache_hit_tokens": 800},
            {**USAGE, "prompt_cache_hit_tokens": 799},
            {**USAGE, "prompt_cache_miss_tokens": -1},
            {**USAGE, "prompt_cache_hit_tokens": 800.5},
        ]
        for usage in inputs:
            with self.subTest(usage=usage):
                record = record_usage(ledger, usage=usage)
                self.assertFalse(record["usage_missing"])
                self.assertTrue(record["estimated_cache"])
                self.assertEqual(record["cache_hit_tokens"], 0)
                self.assertEqual(record["cache_miss_tokens"], 1000)
                self.assertAlmostEqual(record["estimated_cost"], 0.00027)
        self.assertTrue(ledger.summary()["estimated_cache"])
        self.assertIn("按输入全部未命中保守估算", format_cost(ledger))

    def test_zero_usage_is_valid_and_distinct_from_missing_usage(self):
        ledger = UsageLedger(pricing=FIXED_PRICING)
        record = record_usage(ledger, usage=dict.fromkeys(USAGE, 0))
        self.assertFalse(record["usage_missing"])
        self.assertFalse(record["estimated_cache"])
        self.assertEqual(record["total_tokens"], 0)
        self.assertEqual(record["estimated_cost"], 0)
        self.assertIn("USD 0.00000000", format_usage(record))
        self.assertEqual(ledger.summary()["missing_usage_requests"], 0)

    def test_utc_peak_boundaries_and_weekends(self):
        ledger = UsageLedger()
        cases = [
            ("2026-09-18T00:59:59Z", "off_peak"),
            ("2026-09-18T01:00:00Z", "peak"),
            ("2026-09-18T03:59:59Z", "peak"),
            ("2026-09-18T04:00:00Z", "off_peak"),
            ("2026-09-18T05:59:59Z", "off_peak"),
            ("2026-09-18T06:00:00Z", "peak"),
            ("2026-09-18T09:59:59Z", "peak"),
            ("2026-09-18T10:00:00Z", "off_peak"),
            ("2026-09-19T02:00:00Z", "off_peak"),
            ("2026-09-20T07:00:00Z", "off_peak"),
            ("2026-09-21T01:00:00Z", "peak"),
        ]
        for date, period in cases:
            with self.subTest(date=date):
                created = created_at(date)
                record = record_usage(ledger, created=created)
                self.assertEqual(record["created"], created)
                self.assertEqual(record["rate_period"], period)
                self.assertAlmostEqual(record["estimated_cost"], 0.0003048 if period == "peak" else 0.0001524)

    def test_mixed_peak_and_off_peak_requests_preserve_their_rates(self):
        ledger = UsageLedger()
        record_usage(ledger, created=created_at("2026-09-18T03:59:59Z"))
        record_usage(ledger, created=created_at("2026-09-18T04:00:00Z"))
        self.assertAlmostEqual(ledger.summary()["estimated_cost"], 0.0004572)
        output = format_cost(ledger)
        for text in ("高峰价", "低谷价", "USD 0.00045720", "按 API 响应创建时间估算", "跨时段费用可能有差异", "仅保存在内存中", "2026-09-18"):
            self.assertIn(text, output)

    def test_mixed_provider_currencies_are_reported_separately(self):
        ledger = UsageLedger(pricing=FIXED_PRICING)
        record_usage(ledger, model="deepseek-test")
        ledger.pricing = {
            "input_hit_per_million": 0,
            "input_miss_per_million": 1,
            "output_per_million": 2,
            "currency": "CNY",
        }
        record_usage(ledger, model="glm-test")
        output = format_cost(ledger)
        self.assertIn("预估费用小计：", output)
        self.assertIn("  USD ", output)
        self.assertIn("  CNY ", output)
        self.assertIn("未跨币种相加", output)
        self.assertNotIn("预估费用合计：", output)

    def test_missing_creation_time_uses_record_time(self):
        timestamp = created_at("2026-09-18T06:00:00Z")
        with patch("harness.usage.time.time", return_value=timestamp):
            record = record_usage(UsageLedger())
        self.assertEqual(record["created"], timestamp)
        self.assertEqual(record["rate_period"], "peak")

    def test_records_are_isolated_snapshots(self):
        ledger = UsageLedger(pricing=FIXED_PRICING)
        returned = record_usage(ledger)
        returned["total_tokens"] = -1
        snapshot = ledger.records
        snapshot[0]["total_tokens"] = -2
        snapshot.clear()
        self.assertEqual(ledger.summary()["total_tokens"], 1200)
        self.assertEqual(ledger.records[0]["total_tokens"], 1200)

    def test_concurrent_records_keep_unique_request_numbers(self):
        ledger = UsageLedger(pricing=FIXED_PRICING)
        with ThreadPoolExecutor(max_workers=4) as pool:
            records = list(pool.map(lambda _: record_usage(ledger), range(200)))
        self.assertEqual(sorted(record["request"] for record in records), list(range(1, 201)))
        self.assertEqual(ledger.summary()["total_tokens"], 240000)
        self.assertIn("模型请求：200 次", format_cost(ledger))

    def test_empty_session_and_invalid_rates(self):
        ledger = UsageLedger()
        self.assertEqual(ledger.summary()["requests"], 0)
        self.assertIn("暂无模型请求", format_cost(ledger))
        for invalid in ({**FIXED_PRICING, "output_per_million": -1}, {**PRICING, "peak": {}}, {**FIXED_PRICING, "output_per_million": True}):
            with self.subTest(pricing=invalid), self.assertRaises(ValueError):
                UsageLedger(pricing=invalid)


if __name__ == "__main__":
    unittest.main()
