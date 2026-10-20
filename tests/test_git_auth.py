from __future__ import annotations

import pytest

from setup_work import harness
from setup_work.config import HarnessSpec
from setup_work.errors import SetupWorkError
from setup_work.runner import Result, Runner


def spec(host="bitbucket.corp", repo="agent"):
    return HarnessSpec.model_validate(
        {
            "source": "git",
            "repository": f"https://{host}/scm/team/{repo}.git",
            "ref": "main",
            "install": {"method": "setup"},
        }
    )


@pytest.fixture
def git_calls(monkeypatch):
    calls, helpers = [], {}

    def run(self, argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[1:4] == ["config", "--global", "--get-all"]:
            values = helpers.get(argv[4], [])
            return Result(argv, 0 if values else 1, "\n".join(values))
        if argv[1:4] == ["config", "--global", "--add"]:
            helpers.setdefault(argv[4], []).append(argv[5])
        return Result(argv, 0)

    monkeypatch.setattr(Runner, "run", run)
    return calls, helpers


def test_interactive_auth_prompts_once_per_host_and_keeps_existing_helpers(cfg, git_calls):
    calls, helpers = git_calls
    key = "credential.https://bitbucket.corp.helper"
    helpers[key] = ["existing-keychain"]
    runner = Runner(interactive=True)
    harness.prepare_git_auth(cfg, spec(), runner)
    harness.prepare_git_auth(cfg, spec(repo="another-agent"), runner)
    auth = [(a, kw) for a, kw in calls if a[1] == "ls-remote"]
    assert len(auth) == 1
    argv, kwargs = auth[0]
    assert argv[-1] == "HEAD"
    assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "1"
    assert kwargs["capture"] is True and kwargs["timeout"] is None
    assert helpers[key][0] == "existing-keychain"
    assert len(helpers[key]) == 2 and "cache --timeout=28800" in helpers[key][1]
    assert str(cfg.runtime_dir / "git-credentials/socket") in helpers[key][1]
    assert (cfg.runtime_dir / "git-credentials").stat().st_mode & 0o077 == 0
    # Новый вызов setup использует тот же helper, не добавляя дубликат.
    harness.prepare_git_auth(cfg, spec(), Runner(interactive=True))
    assert len(helpers[key]) == 2


def test_another_host_gets_its_own_login(cfg, git_calls):
    calls, helpers = git_calls
    runner = Runner(interactive=True)
    for host in ("one.corp", "two.corp", "one.corp"):
        harness.prepare_git_auth(cfg, spec(host), runner)
    assert len([a for a, kw in calls if a[1] == "ls-remote"]) == 2
    assert len(helpers) == 2


@pytest.mark.parametrize("dry_run,interactive", [(True, True), (False, False)])
def test_no_login_or_config_changes_without_tty_or_during_dry_run(
    cfg, git_calls, dry_run, interactive
):
    calls, helpers = git_calls
    harness.prepare_git_auth(cfg, spec(), Runner(dry_run=dry_run, interactive=interactive))
    assert not calls and not helpers and not cfg.runtime_dir.exists()


def test_ssh_auth_is_left_to_existing_ssh_configuration(cfg, git_calls):
    calls, _ = git_calls
    ssh = spec().model_copy(update={"repository": "ssh://git@bitbucket.corp/team/agent.git"})
    harness.prepare_git_auth(cfg, ssh, Runner(interactive=True))
    assert not calls


def test_failed_auth_is_retryable(cfg, monkeypatch):
    runner = Runner(interactive=True)

    def run(argv, **kwargs):
        if argv[1] == "ls-remote":
            raise SetupWorkError("authentication failed")
        return Result(argv, 1)

    monkeypatch.setattr(runner, "run", run)
    for _ in range(2):
        with pytest.raises(SetupWorkError, match="authentication failed"):
            harness.prepare_git_auth(cfg, spec(), runner)
        assert not runner.authenticated_git_hosts
