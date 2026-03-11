"""Tests for the tribal_village CLI (typer app)."""

from typer.testing import CliRunner

from tests.conftest import requires_nim_library
from tribal_village_env.cli import app

runner = CliRunner()


class TestAppCreation:
    """Verify the Typer app can be created without errors."""

    def test_app_exists(self):
        assert app is not None

    def test_app_has_play_command(self):
        command_names = [cmd.name for cmd in app.registered_commands]
        assert "play" in command_names


class TestPlayHelp:
    """Verify --help flag works."""

    def test_play_help(self):
        result = runner.invoke(app, ["play", "--help"])
        assert result.exit_code == 0
        assert "render" in result.output.lower()

    def test_root_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0


class TestInvalidRenderMode:
    """Verify invalid render mode produces error exit."""

    def test_invalid_render_mode(self):
        result = runner.invoke(app, ["play", "--render", "invalid"])
        assert result.exit_code != 0


@requires_nim_library
class TestPlayAnsi:
    """Integration tests for play command with ANSI renderer."""

    def test_play_ansi_short(self):
        result = runner.invoke(app, ["play", "--render", "ansi", "--steps", "2"])
        assert result.exit_code == 0
        assert len(result.output) > 0
