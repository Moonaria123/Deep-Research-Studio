"""Keep regression tests independent of the operator's provider credentials."""
import os
import secrets

import pytest

_PREFIXES = ('DEEPRESEARCH_', 'OPENAI_', 'ANTHROPIC_', 'BRAVE_', 'FIRECRAWL_', 'TAVILY_', 'EXA_', 'POE_', 'VOLCENGINE_', 'ARK_')


def _provider_variables():
    return [name for name in os.environ if name.startswith(_PREFIXES)]


# Collection must not import an operator's live credentials either.
for _name in _provider_variables():
    os.environ.pop(_name, None)


@pytest.fixture(autouse=True)
def isolated_provider_environment(monkeypatch, tmp_path_factory: pytest.TempPathFactory):
    for name in _provider_variables():
        monkeypatch.delenv(name, raising=False)
    if os.name != "nt":
        key_path = tmp_path_factory.mktemp("isolated-vault-key") / "vault-root.key"
        key_path.write_text(secrets.token_hex(32) + "\n", encoding="ascii")
        key_path.chmod(0o600)
        monkeypatch.setenv("DEEPRESEARCH_VAULT_KEY_FILE", str(key_path))
