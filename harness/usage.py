"""Track per-request token usage and estimated session costs in memory."""

from datetime import datetime, timezone
import math
from threading import Lock
import time


PRICING = {
    "input_hit_per_million": 0.003,
    "input_miss_per_million": 0.15,
    "output_per_million": 0.6,
    "peak": {
        "input_hit_per_million": 0.006,
        "input_miss_per_million": 0.30,
        "output_per_million": 1.2,
    },
    "currency": "USD",
    "source": "https://api-docs.deepseek.com/quick_start/pricing",
    "checked_at": "2026-09-18",
}

_TOKEN_FIELDS = (
    "prompt_tokens", "completion_tokens", "total_tokens",
    "cache_hit_tokens", "cache_miss_tokens",
)
_RATE_FIELDS = (
    "input_hit_per_million", "input_miss_per_million", "output_per_million",
)


def _is_token_count(value):
    return type(value) is int and 0 <= value <= 2**53 - 1


def _valid_rates(rates):
    return isinstance(rates, dict) and all(
        type(rates.get(key)) in (int, float)
        and math.isfinite(rates[key]) and rates[key] >= 0
        for key in _RATE_FIELDS
    )


def _creation_time(created):
    if type(created) in (int, float) and math.isfinite(created):
        try:
            return created, datetime.fromtimestamp(created, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            pass
    timestamp = time.time()
    return timestamp, datetime.fromtimestamp(timestamp, tz=timezone.utc)


def _rate_period(date, has_peak):
    if not has_peak:
        return "fixed"
    peak = date.weekday() < 5 and (1 <= date.hour < 4 or 6 <= date.hour < 10)
    return "peak" if peak else "off_peak"


def _summarize(records):
    summary = {
        "requests": len(records),
        **dict.fromkeys(_TOKEN_FIELDS, 0),
        "estimated_cost": 0.0,
        "estimated_cache": False,
        "missing_usage_requests": 0,
    }
    for record in records:
        if record["usage_missing"]:
            summary["missing_usage_requests"] += 1
        else:
            for key in _TOKEN_FIELDS:
                summary[key] += record[key]
            summary["estimated_cost"] += record["estimated_cost"]
        summary["estimated_cache"] |= record["estimated_cache"]
    return summary


class UsageLedger:
    def __init__(self, pricing=PRICING):
        if not _valid_rates(pricing) or ("peak" in pricing and not _valid_rates(pricing["peak"])):
            raise ValueError("Token 费率必须是有限的非负数字。")
        self.pricing = dict(pricing)
        if "peak" in pricing:
            self.pricing["peak"] = dict(pricing["peak"])
        self._records = []
        self._lock = Lock()

    @property
    def records(self):
        with self._lock:
            return [dict(record) for record in self._records]

    def record(self, *, usage=None, model, turn, created=None):
        timestamp, date = _creation_time(created)
        rate_period = _rate_period(date, "peak" in self.pricing)
        rates = self.pricing["peak"] if rate_period == "peak" else self.pricing
        record = {
            "turn": turn,
            "model": model,
            **dict.fromkeys(_TOKEN_FIELDS),
            "estimated_cost": None,
            "estimated_cache": True,
            "usage_missing": True,
            "rate_period": rate_period,
            "created": timestamp,
            "currency": self.pricing.get("currency", "USD"),
        }
        if (isinstance(usage, dict)
                and all(_is_token_count(usage.get(key)) for key in _TOKEN_FIELDS[:3])
                and usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]):
            cache_valid = (
                _is_token_count(usage.get("prompt_cache_hit_tokens"))
                and _is_token_count(usage.get("prompt_cache_miss_tokens"))
                and usage["prompt_cache_hit_tokens"] + usage["prompt_cache_miss_tokens"]
                == usage["prompt_tokens"]
            )
            for key in _TOKEN_FIELDS[:3]:
                record[key] = usage[key]
            record["cache_hit_tokens"] = usage["prompt_cache_hit_tokens"] if cache_valid else 0
            record["cache_miss_tokens"] = usage["prompt_cache_miss_tokens"] if cache_valid else usage["prompt_tokens"]
            record["estimated_cost"] = (
                record["cache_hit_tokens"] * rates["input_hit_per_million"]
                + record["cache_miss_tokens"] * rates["input_miss_per_million"]
                + record["completion_tokens"] * rates["output_per_million"]
            ) / 1_000_000
            record["estimated_cache"] = not cache_valid
            record["usage_missing"] = False
        with self._lock:
            record["request"] = len(self._records) + 1
            self._records.append(record)
        return dict(record)

    def summary(self):
        return _summarize(self.records)


def format_usage(record):
    prefix = f"[请求 #{record['request']}／对话 {record['turn']}]"
    if record["usage_missing"]:
        return f"{prefix} 用量缺失或无效，token 与费用未知。"
    notes = []
    if record["rate_period"] == "peak":
        notes.append("高峰价")
    elif record["rate_period"] == "off_peak":
        notes.append("低谷价")
    if record["estimated_cache"]:
        notes.append("缓存按全未命中估算")
    suffix = f"（{'；'.join(notes)}）" if notes else ""
    return (
        f"{prefix} 输入 {record['prompt_tokens']}，输出 {record['completion_tokens']}，"
        f"合计 {record['total_tokens']} token；预估费用 "
        f"{record.get('currency', 'USD')} {record['estimated_cost']:.8f}{suffix}。"
    )


def _format_rates(label, rates, currency):
    return (
        f"{label}：输入缓存命中 {rates['input_hit_per_million']:.8f}，"
        f"输入缓存未命中 {rates['input_miss_per_million']:.8f}，"
        f"输出 {rates['output_per_million']:.8f} {currency}／百万 token。"
    )


def format_cost(ledger):
    records = ledger.records
    summary = _summarize(records)
    pricing = ledger.pricing
    currency = pricing.get("currency", "USD")
    subtotal = "已知小计" if summary["missing_usage_requests"] else "合计"
    lines = [
        "本次会话用量与费用（仅保存在内存中，退出后不保留）",
        f"模型请求：{summary['requests']} 次。",
        f"Token {subtotal}：输入 {summary['prompt_tokens']}，输出 {summary['completion_tokens']}，合计 {summary['total_tokens']}。",
        f"输入缓存{subtotal}：命中 {summary['cache_hit_tokens']}，未命中 {summary['cache_miss_tokens']}。",
        f"预估费用{subtotal}：{currency} {summary['estimated_cost']:.8f}，以实际账单为准。",
    ]
    if summary["missing_usage_requests"]:
        lines.append(
            f"{summary['missing_usage_requests']} 次请求用量缺失或无效，未计入小计；这些请求的 token 与费用未知。"
        )
    if any(record["estimated_cache"] and not record["usage_missing"] for record in records):
        lines.append("部分请求的缓存字段缺失或不一致，按输入全部未命中保守估算。")
    if "peak" in pricing:
        lines.extend([
            _format_rates("低谷价", pricing, currency),
            _format_rates("高峰价", pricing["peak"], currency),
            "高峰时段：UTC 周一至周五 01:00–04:00、06:00–10:00（均不含结束时刻），其余为低谷。",
            "按 API 响应创建时间估算费率，缺失时使用本地记录时间；官方未说明跨时段请求的计费时点，跨时段费用可能有差异。",
        ])
    else:
        lines.append(_format_rates("费率", pricing, currency))
    if pricing.get("source"):
        lines.append(f"费率来源：{pricing['source']}")
    if pricing.get("checked_at"):
        lines.append(f"费率核验日期：{pricing['checked_at']}。")
    lines.append("逐请求明细：")
    lines.extend(map(format_usage, records) if records else ["暂无模型请求。"])
    return "\n".join(lines)
