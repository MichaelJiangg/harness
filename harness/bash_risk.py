"""Conservative shell heuristics; this is not a shell parser or a sandbox."""

import posixpath
import re
import shlex


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
_SYSTEM_COMMANDS = {
    "ls": "ls", "/bin/ls": "ls", "/usr/bin/ls": "ls",
    "pwd": "pwd", "/bin/pwd": "pwd", "/usr/bin/pwd": "pwd",
}
_LS_SHORT = frozenset("aAbBcCdFfghHiIklmnpqQrRsStTuUvwx1")
_LS_LONG = frozenset((
    "--all", "--almost-all", "--directory", "--human-readable", "--inode",
    "--numeric-uid-gid", "--recursive", "--reverse", "--size",
    "--group-directories-first", "--color=never", "--color=auto", "--color=always",
))


def classify_bash_risk(command: str) -> str:
    """Only recognized ls/pwd commands are read-only; unknown syntax needs approval."""
    if not isinstance(command, str) or not command.strip():
        return "write"
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        tokens = []
    # Inspect both spelling and unquoted tokens, including quoted command names.
    for text in (command, " ".join(tokens)):
        if _PIPE_EXEC.search(text) or any(pattern.search(text) for pattern in _DESTRUCTIVE):
            return "destructive"
    # Do not infer safety for expansions, redirects, jobs, scripts, or shell options.
    if not tokens or any(char in command for char in "$`<>;#\n\r(){}*?[]\\\x00"):
        return "write"
    parts = [[]]
    for token in tokens:
        if token in {"&&", "||"}:
            parts.append([])
        elif token in {"&", "|"} or set(token) <= {"&", "|", ";"}:
            return "write"
        else:
            parts[-1].append(token)
    return max((_simple_risk(part) for part in parts), key=_LEVELS.__getitem__)


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
