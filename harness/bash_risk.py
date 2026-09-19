"""Conservative shell heuristics; this is not a shell parser or a sandbox."""

import posixpath
import re
import shlex
from pathlib import Path


_LEVELS = {"read_only": 0, "write": 1, "destructive": 2}
_DESTRUCTIVE = tuple(re.compile(pattern) for pattern in (
    r"\brm\s+(?:-[^\s;&|]+\s+)*(?:--recursive\b|-[A-Za-z]*[rR][A-Za-z]*\b)",
    r"\bmkfs(?:\.[A-Za-z0-9]+)?\b",
    r"\bdd\s+[^;&|]*\bif\s*=",
    r"\bchmod\s+(?=[^;&|]*(?:-[A-Za-z]*R|--recursive))[^;&|]*\b0?777\b",
    r">\s*/dev/(?:sd|hd|vd|nvme|disk|rdisk)",
    r"\bgit\s+[^;&|]*\bpush\b[^;&|]*(?:--force(?:-with-lease|-if-includes)?\b|(?:^|\s)-[A-Za-z]*f\b)",
))
_PIPE_EXEC = re.compile(
    r"\b(?:curl|wget)\b[^;\n]*\|\s*(?:(?:/[^\s|]+/)?(?:bash|sh|zsh))\b"
)
_SENSITIVE = re.compile(
    r"(?:^|[\s'\"=])(?:/etc|/usr|/System|/root|/dev|~/\.ssh)(?:/|$|[\s'\"])",
)
_AUTO_DANGEROUS = re.compile(
    r"\b(?:curl|wget|ssh|scp|rsync|git\s+push|git\s+fetch|kill|pkill|killall|"
    r"chmod|chown|sudo|mkfs|dd|shutdown|reboot|nohup|npm\s+(?:install|ci|audit|publish|add))\b"
)
_SYSTEM_COMMANDS = {
    "ls": "ls", "/bin/ls": "ls", "/usr/bin/ls": "ls",
    "pwd": "pwd", "/bin/pwd": "pwd", "/usr/bin/pwd": "pwd",
    "cat": "cat", "/bin/cat": "cat", "/usr/bin/cat": "cat",
    "head": "head", "/usr/bin/head": "head", "/bin/head": "head",
    "tail": "tail", "/usr/bin/tail": "tail", "/bin/tail": "tail",
    "sed": "sed", "/usr/bin/sed": "sed", "/bin/sed": "sed",
    "grep": "grep", "/usr/bin/grep": "grep", "/bin/grep": "grep",
    "find": "find", "/usr/bin/find": "find", "/bin/find": "find",
    "wc": "wc", "/usr/bin/wc": "wc", "/bin/wc": "wc",
    "echo": "echo", "/bin/echo": "echo", "/usr/bin/echo": "echo",
    "printf": "printf", "/usr/bin/printf": "printf", "/bin/printf": "printf",
    "cd": "cd",
    "node": "node", "python": "python", "python3": "python3",
}
_LS_SHORT = frozenset("aAbBcCdFfghHiIklmnpqQrRsStTuUvwx1")
_LS_LONG = frozenset((
    "--all", "--almost-all", "--directory", "--human-readable", "--inode",
    "--numeric-uid-gid", "--recursive", "--reverse", "--size",
    "--group-directories-first", "--color=never", "--color=auto", "--color=always",
))


def classify_bash_risk(command: str) -> str:
    """Only conservative read-only inspection commands can bypass confirmation."""
    if not isinstance(command, str) or not command.strip():
        return "write"
    raw_command = command
    # Discard only the exact stderr-null redirect so version checks and harmless
    # inspection commands are not treated as arbitrary shell redirection.
    command = re.sub(r"\b2\s*>\s*/dev/null\b", "", command)
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        tokens = []
    # Inspect both spelling and unquoted tokens, including quoted command names.
    for text in (raw_command, " ".join(tokens)):
        if _PIPE_EXEC.search(text) or any(pattern.search(text) for pattern in _DESTRUCTIVE):
            return "destructive"
    # Do not infer safety for expansions, redirects, jobs, scripts, or shell options.
    if not tokens or any(char in command for char in "$`<>#\n\r(){}\\\x00"):
        return "write"
    parts = [[]]
    for token in tokens:
        if token in {"&&", "||"}:
            parts.append([])
        elif token in {"&", "|"}:
            return "write"
        elif token == ";":
            parts.append([])
        else:
            parts[-1].append(token)
    return max((_simple_risk(part) for part in parts), key=_LEVELS.__getitem__)


def bash_auto_allowed(command, workspace, auto_directories=()):
    """Auto 模式只排除明确危险，不假装能理解任意 Shell 语义。"""
    if not isinstance(command, str) or not command.strip():
        return False
    if classify_bash_risk(command) == "destructive":
        return False
    if _SENSITIVE.search(command) or _AUTO_DANGEROUS.search(command):
        return False
    if ".." in command or "$(" in command or "`" in command:
        return False
    inspection = re.sub(r"\b2\s*>\s*/dev/null\b", "", command)
    if any(character in inspection for character in "<>"):
        return False
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return False
    if not tokens or any(char in command for char in "(){}"):
        return False

    root = Path(workspace).resolve()
    roots = [root] if not auto_directories else [
        (root / directory).resolve() for directory in auto_directories
    ]
    parts = [[]]
    for token in tokens:
        if token in {"&&", ";", "||"}:
            parts.append([])
        elif token in {"&", "|"}:
            return False
        else:
            parts[-1].append(token)
    current = root if any(root.is_relative_to(allowed) for allowed in roots) else None
    for part in parts:
        if not part:
            continue
        if part[0] == "cd":
            if len(part) < 2:
                continue
            candidate = (root / part[1]).resolve()
            if not any(candidate.is_relative_to(allowed) for allowed in roots):
                return False
            current = candidate
        elif current is None:
            return False
    return True


def _simple_risk(tokens):
    if not tokens or tokens[0] not in _SYSTEM_COMMANDS:
        # Includes assignments, export, interpreters, wrappers, and unknown tools.
        return "write"
    name, arguments = _SYSTEM_COMMANDS[tokens[0]], tokens[1:]
    if any(
        _SENSITIVE.search(argument)
        or _SENSITIVE.search(posixpath.normpath(argument).replace("//", "/"))
        or ".ssh" in argument.split("/")
        for argument in arguments
    ):
        return "write"
    if name == "pwd":
        return "read_only" if all(arg in {"-L", "-P"} for arg in arguments) else "write"
    if name == "cd":
        return "read_only" if all(not arg.startswith("-") for arg in arguments) else "write"
    if name in {"echo", "printf", "wc", "grep"}:
        return "read_only"
    if name == "find":
        dangerous = {"-delete", "-exec", "-execdir", "-ok", "-okdir",
                     "-fprint", "-fprint0", "-fprintf", "-fls"}
        return "write" if dangerous.intersection(arguments) else "read_only"
    if name in {"node", "python", "python3"}:
        return "read_only" if arguments and set(arguments) <= {"-v", "--version", "-V"} else "write"
    if name == "sed":
        if "-i" in arguments or "--in-place" in arguments:
            return "write"
        if not any(argument in {"-n", "--quiet", "--silent"} for argument in arguments):
            return "write"
        positional = [argument for argument in arguments if not argument.startswith("-")]
        if not positional or "w" in positional[0]:
            return "write"
        return "read_only"
    if name in {"cat", "head", "tail"}:
        allowed = {
            "cat": {"-n", "-b", "-s", "-v", "-E", "-T",
                    "--number", "--number-nonblank", "--squeeze-blank"},
            "head": {"-n", "-c", "-q", "-v", "--lines", "--bytes", "--quiet", "--verbose"},
            "tail": {"-n", "-c", "-q", "-v", "--lines", "--bytes", "--quiet", "--verbose"},
        }[name]
        for argument in arguments:
            if argument in {"-f", "-F", "--follow"}:
                return "write"
            if name in {"head", "tail"} and re.fullmatch(r"-\d+", argument):
                continue
            if argument.startswith("-") and argument not in allowed:
                return "write"
        return "read_only"
    options = True
    for argument in arguments:
        if options and argument == "--":
            options = False
        elif options and argument.startswith("--"):
            if argument not in _LS_LONG:
                return "write"
        elif options and argument.startswith("-") and argument != "-":
            if not set(argument[1:]) <= _LS_SHORT:
                return "write"
    return "read_only"
