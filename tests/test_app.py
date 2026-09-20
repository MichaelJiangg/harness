import unittest
from unittest.mock import Mock, patch

from harness import app
from harness.app import create_app


class AppAssemblyTests(unittest.TestCase):
    def test_create_app_returns_callable_assembled_entry(self):
        client = Mock(provider="deepseek")
        settings = {
            "provider": "auto",
            "model": {"name": "deepseek-flash"},
            "glm": {"name": "glm-5.3-flash"},
            "mcp": {"enabled": True},
        }
        with patch.object(app, "get_settings", return_value=settings) as get_settings, \
                patch.object(app, "select_model_provider", return_value=("deepseek", "test-key")) as select, \
                patch.object(app, "ChatCompletionClient", return_value=client) as client_factory, \
                patch.object(app, "run_cli") as run_cli:
            start = create_app()
            start()
        get_settings.assert_called_once_with()
        select.assert_called_once_with()
        client_factory.assert_called_once_with(
            "test-key", provider="deepseek", model="deepseek-flash",
        )
        self.assertTrue(callable(start))
        self.assertIs(start.client, client)
        self.assertEqual(start.provider, "deepseek")
        self.assertIs(start.settings, settings)
        run_cli.assert_called_once_with(
            client,
            ledger=None,
            input_stream=None,
            output=None,
            error_output=None,
            character_delay=None,
            memory_enabled=True,
            memory_store=None,
            notes_enabled=True,
            notes_store=None,
            vector_store=None,
            hooks_enabled=True,
            hooks_manager=None,
            mcp_enabled=True,
        )

    def test_create_app_accepts_injected_client_without_loading_keys(self):
        client = Mock(provider="glm")
        with patch.object(app, "select_model_provider") as select, \
                patch.object(app, "ChatCompletionClient") as client_factory, \
                patch.object(app, "run_cli") as run_cli:
            create_app(client=client)()
        select.assert_not_called()
        client_factory.assert_not_called()
        self.assertIs(run_cli.call_args.args[0], client)

    def test_only_create_app_is_exported(self):
        self.assertEqual(app.__all__, ["create_app"])


if __name__ == "__main__":
    unittest.main()
