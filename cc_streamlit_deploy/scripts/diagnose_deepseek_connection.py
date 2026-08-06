"""Read-only DeepSeek connection diagnostics.

The report contains no API key or request payload. It is safe to run locally
and is intentionally independent of Streamlit state.
"""

from __future__ import annotations

import socket
import ssl
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api_keys import get_deepseek_settings
from src.deepseek_client import DeepSeekClient, DeepSeekRequestError


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "deepseek_connection_diagnostic.md"


def mask_key(value: str | None) -> str:
    if not value:
        return "absent"
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}...{value[-2:]}"


def mask_url(value: str) -> str:
    from urllib.parse import urlsplit

    parsed = urlsplit(value)
    return f"{parsed.scheme}://{parsed.netloc}/..." if parsed.netloc else "<empty>"


def network_probe(base_url: str) -> dict[str, str]:
    from urllib.parse import urlsplit

    parsed = urlsplit(base_url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    result: dict[str, str] = {}
    started = time.perf_counter()
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        result["DNS"] = f"OK ({len(addresses)} addresses, {time.perf_counter() - started:.3f}s)"
    except OSError as exc:
        result["DNS"] = f"FAIL ({type(exc).__name__})"
        return result
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=5) as raw:
            result["TCP"] = f"OK ({time.perf_counter() - started:.3f}s)"
            if parsed.scheme == "https":
                context = ssl.create_default_context()
                with context.wrap_socket(raw, server_hostname=host):
                    result["TLS"] = f"OK ({time.perf_counter() - started:.3f}s)"
    except TimeoutError:
        result["TCP/TLS"] = "FAIL (timeout)"
    except (OSError, ssl.SSLError) as exc:
        result["TCP/TLS"] = f"FAIL ({type(exc).__name__})"
    return result


def main() -> int:
    settings = get_deepseek_settings()
    endpoint = f"{settings.base_url.rstrip('/')}/chat/completions"
    proxies = requests.utils.get_environ_proxies(endpoint)
    env_proxy_names = {
        name: "set" if __import__("os").environ.get(name) else "unset"
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
    }
    lines = [
        "# DeepSeek连接诊断",
        "",
        f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- model：`{settings.model}`",
        f"- base_url（掩码）：`{mask_url(settings.base_url)}`",
        f"- API Key：{'存在' if settings.api_key else '不存在'}；长度 `{len(settings.api_key or '')}`；掩码 `{mask_key(settings.api_key)}`",
        f"- 配置来源：`{settings.source}`",
        "",
        "## 代理检查",
        "",
    ]
    lines.extend(f"- {key}：{value}" for key, value in env_proxy_names.items())
    lines.append(f"- requests解析到的系统代理：`{'存在' if proxies else '不存在'}`")
    if proxies:
        from urllib.parse import urlsplit
        hosts = sorted({urlsplit(str(value)).netloc or '<unknown>' for value in proxies.values()})
        lines.append(f"- 代理主机（掩码）：`{', '.join(hosts)}`")
    lines += ["", "## DNS/TLS/连接", ""]
    lines.extend(f"- {key}：{value}" for key, value in network_probe(settings.base_url).items())
    lines += ["", "## 最小请求（仅回复OK）", ""]
    if not settings.api_key:
        lines.append("- 未执行：缺少API Key。")
    else:
        for index in range(1, 4):
            client = DeepSeekClient(settings)
            started = time.perf_counter()
            try:
                answer = client.chat(
                    [{"role": "user", "content": "仅回复OK"}],
                    max_tokens=8,
                    connect_timeout=5,
                    timeout=10,
                    max_retries=3,
                )
                lines.append(
                    f"- 第{index}次：成功；响应=`{answer[:20]}`；耗时 `{time.perf_counter() - started:.3f}s`；"
                    f"请求次数 `{client.usage_snapshot()['api_call_count']}`；重试 `{client.usage_snapshot()['retry_count']}`；"
                    f"代理路径=`{'直连（绕过环境代理）' if client._bypass_broken_proxy else '环境代理/直连'}`"
                )
            except DeepSeekRequestError as exc:
                lines.append(
                    f"- 第{index}次：失败；分类=`{exc.category}`；HTTP=`{exc.status_code or '无'}`；"
                    f"耗时 `{exc.elapsed_seconds or round(time.perf_counter() - started, 3)}s`；重试 `{exc.retry_count}`；"
                    f"代理路径=`{'直连' if exc.proxy_used is False else '环境代理或未知'}`；错误=`{exc}`"
                )
            except Exception as exc:
                lines.append(f"- 第{index}次：失败；分类=`{type(exc).__name__}`；错误=`{exc}`")
    lines += [
        "",
        "## 判断",
        "",
        "诊断脚本只读取现有配置，不修改环境变量、代理设置或项目数据。",
        "若环境代理失败而直连成功，说明请求路径受Windows用户级代理污染；这不等同于CC Switch/Codex线路故障。",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
