"""读取启动密钥，不修改进程环境或执行 .env 内容。"""

import os
from pathlib import Path
import shlex

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_api_key(*, env_file=None, environ=None):
    environ = os.environ if environ is None else environ
    if "DEEPSEEK_API_KEY" in environ:
        return environ["DEEPSEEK_API_KEY"]

    path = ENV_FILE if env_file is None else Path(env_file)
    try:
        content = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError):
        raise ValueError("无法读取项目 .env，请检查文件权限和 UTF-8 编码。") from None

    api_key = None
    for number, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        if name.strip() != "DEEPSEEK_API_KEY":
            continue
        try:
            parts = shlex.split(raw_value, comments=True, posix=True)
            if not separator or len(parts) > 1:
                raise ValueError
        except ValueError:
            raise ValueError(f".env 第 {number} 行的 DEEPSEEK_API_KEY 格式无效，请检查引号与赋值格式。") from None
        api_key = parts[0] if parts else ""
    return api_key
