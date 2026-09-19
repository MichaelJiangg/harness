"""项目根目录长期笔记：HARNESS.md 的读取、注入和原子更新。"""

import os
from pathlib import Path
from stat import S_IMODE, S_ISREG
from tempfile import NamedTemporaryFile
from threading import RLock


NOTES_FILE_NAME = "HARNESS.md"
MAX_NOTES_BYTES = 65536
MAX_CONTEXT_NOTES_CHARS = 8000


class NotesError(RuntimeError):
    """可安全回传给模型或终端的笔记错误。"""


def _clean_context(value):
    return "".join(
        character for character in value
        if character in "\n\t" or character.isprintable()
    )


def notes_system_prompt(base_prompt, content):
    if not content.strip():
        return base_prompt
    content = _clean_context(content)
    if len(content) > MAX_CONTEXT_NOTES_CHARS:
        head = content[:6000].rstrip()
        tail = content[-2000:].lstrip()
        content = (
            head
            + "\n\n<!-- 中间内容已省略，可用 notes_read 查看完整项目笔记 -->\n\n"
            + tail
        )
    context = (
        "## 项目长期笔记（HARNESS.md）\n"
        "以下是项目根目录长期笔记，记录技术栈、编码约定、架构决定、已知问题和当前进展。"
        "进行项目工作时遵守其中明确适用的约定；用户当前明确指令与之冲突时，以当前指令为准。\n"
        + content
    )
    return base_prompt + "\n\n" + context


class NotesStore:
    """只操作工作区根目录的单个 Markdown 笔记文件。"""

    def __init__(self, workspace=None, *, max_bytes=MAX_NOTES_BYTES):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("笔记字节上限必须是正整数。")
        self.workspace = (Path.cwd() if workspace is None else Path(workspace)).resolve()
        self.path = self.workspace / NOTES_FILE_NAME
        self.max_bytes = max_bytes
        self._lock = RLock()

    def read(self):
        with self._lock:
            try:
                if not self.path.exists():
                    return ""
                if self.path.is_symlink():
                    raise NotesError("HARNESS.md 是软链接，已拒绝读取。") from None
                metadata = self.path.stat()
                if not S_ISREG(metadata.st_mode):
                    raise NotesError("HARNESS.md 不是普通文件，已拒绝读取。") from None
                if metadata.st_size > self.max_bytes:
                    raise NotesError(
                        f"HARNESS.md 超过 {self.max_bytes} 字节上限，请先缩小文件。"
                    ) from None
                data = self.path.read_bytes()
                if len(data) > self.max_bytes:
                    raise NotesError(
                        f"HARNESS.md 超过 {self.max_bytes} 字节上限，请先缩小文件。"
                    ) from None
                content = data.decode("utf-8")
                if "\x00" in content:
                    raise NotesError("HARNESS.md 包含二进制内容，仅支持 UTF-8 文本。") from None
                return content
            except NotesError:
                raise
            except FileNotFoundError:
                return ""
            except PermissionError:
                raise NotesError("没有读取 HARNESS.md 的权限。") from None
            except UnicodeDecodeError:
                raise NotesError("HARNESS.md 不是有效的 UTF-8 文本。") from None
            except (OSError, RuntimeError, ValueError):
                raise NotesError("无法读取 HARNESS.md，请检查文件状态。") from None

    def append(self, content):
        content = self._validate_content(content)
        with self._lock:
            current = self.read()
            proposed = self._append_markdown(current, content)
            if len(proposed.encode("utf-8")) > self.max_bytes:
                raise NotesError(
                    f"追加后 HARNESS.md 将超过 {self.max_bytes} 字节上限。"
                )
            self._write(proposed)
        return len(content.encode("utf-8"))

    def replace(self, content):
        content = self._validate_content(content, allow_empty=True)
        with self._lock:
            if len(content.encode("utf-8")) > self.max_bytes:
                raise NotesError(
                    f"内容超过 {self.max_bytes} 字节上限，请先缩小内容。"
                )
            self._write(content)
        return len(content.encode("utf-8"))

    def clear(self):
        return self.replace("")

    def _validate_content(self, content, *, allow_empty=False):
        if not isinstance(content, str):
            raise NotesError("笔记内容必须是字符串。")
        if "\x00" in content:
            raise NotesError("笔记内容包含空字符，仅支持 UTF-8 文本。")
        if not content.strip() and not allow_empty:
            raise NotesError("待追加的笔记内容不能为空。")
        return content

    @staticmethod
    def _append_markdown(current, content):
        content = content.strip("\n").rstrip()
        if not current:
            return content + "\n"
        previous = current.rstrip("\n").splitlines()[-1] if current.rstrip("\n") else ""
        bullet_start = ("- ", "* ", "+ ")
        if (previous.lstrip().startswith(bullet_start)
                and content.lstrip().startswith(bullet_start)):
            return current.rstrip("\n") + "\n" + content + "\n"
        return current.rstrip("\n") + "\n\n" + content + "\n"

    def _write(self, content):
        temporary = None
        try:
            if self.path.exists():
                if self.path.is_symlink():
                    raise NotesError("HARNESS.md 是软链接，已拒绝更新。") from None
                metadata = self.path.stat()
                if not S_ISREG(metadata.st_mode):
                    raise NotesError("HARNESS.md 不是普通文件，已拒绝更新。") from None
                mode = S_IMODE(metadata.st_mode)
            else:
                mode = 0o644
            with NamedTemporaryFile(
                "wb", dir=self.workspace, prefix=".harness-notes-", delete=False,
            ) as target:
                temporary = Path(target.name)
                target.write(content.encode("utf-8"))
            os.chmod(temporary, mode)
            if self.path.is_symlink():
                raise NotesError("HARNESS.md 已变成软链接，未执行更新。") from None
            temporary.replace(self.path)
        except NotesError:
            raise
        except PermissionError:
            raise NotesError("没有更新 HARNESS.md 的权限。") from None
        except UnicodeEncodeError:
            raise NotesError("笔记内容无法编码为有效的 UTF-8 文本。") from None
        except (OSError, RuntimeError, ValueError):
            raise NotesError("无法更新 HARNESS.md，请检查文件状态。") from None
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
