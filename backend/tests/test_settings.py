from pathlib import Path
from tempfile import TemporaryDirectory

from kb_web_agent.settings import load_settings, normalize_path


def test_windows_knowledge_base_path_normalizes_to_wsl_mount():
    assert normalize_path(r"D:\workNote") == Path("/mnt/d/workNote")


def test_settings_loads_from_env_file_without_mutating_process_env(monkeypatch):
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    with TemporaryDirectory() as directory:
        env_path = Path(directory) / ".env"
        env_path.write_text(
            "\n".join(
                [
                    "ARK_API_KEY=test-key",
                    "ARK_MODEL=glm-5.1",
                    "KNOWLEDGE_BASE_PATH=D:\\workNote",
                ]
            ),
            encoding="utf-8",
        )

        settings = load_settings(env_file=env_path)

    assert settings.api_key == "test-key"
    assert settings.model == "glm-5.1"
    assert settings.knowledge_base_path == Path("/mnt/d/workNote")
