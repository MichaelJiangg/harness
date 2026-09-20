from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from harness.presets import format_written_file


class PresetTests(unittest.TestCase):
    def test_python_file_uses_ruff_when_available(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "sample.py"
            target.write_text("x=1\n", encoding="utf-8")
            with patch("harness.presets.shutil.which", return_value="/usr/local/bin/ruff"), \
                    patch("harness.presets.subprocess.run", return_value=SimpleNamespace(
                        returncode=0, stderr="",
                    )) as run:
                formatter = format_written_file(root, "sample.py")
            self.assertEqual(formatter, "ruff")
            self.assertEqual(run.call_args.args[0][0], "ruff")

    def test_no_formatter_returns_none_without_running_commands(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.py").write_text("x=1\n", encoding="utf-8")
            with patch("harness.presets.shutil.which", return_value=None), \
                    patch("harness.presets.subprocess.run") as run:
                self.assertIsNone(format_written_file(root, "sample.py"))
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
