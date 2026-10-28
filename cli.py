"""``hermes recall`` terminal subcommands, discovered by the Hermes memory loader.

``discover_plugin_cli_commands`` (``plugins/memory/__init__.py``) looks for this
file at ``<provider dir>/cli.py`` — the repository root for a directory install —
imports it as ``_hermes_user_memory.recall.cli``, calls ``register_cli(subparser)``
at argparse setup time and picks up ``recall_command`` as the handler (it looks
for ``<provider name>_command``). Only the active ``memory.provider`` gets its CLI.

Imports of the package are lazy and inside the functions: the loader imports this
file during argparse setup, before any provider is loaded, and also executes it as
a submodule when it loads the provider directory, so module scope must stay cheap
and import-safe.

Commands:

* ``hermes recall status`` — the same status block as ``/recall``.
* ``hermes recall show <id>`` — the same detail view as ``/recall <id>``.
"""

from __future__ import annotations

from typing import Any


def _package() -> Any:
    """Return the ``hermes_recall`` package via the loader route or a flat import."""
    try:  # loaded by the Hermes memory loader as _hermes_user_memory.recall.cli
        from . import hermes_recall as package  # type: ignore[attr-defined]

        return package
    except ImportError:  # imported flat (repository root on sys.path)
        import hermes_recall as package

        return package


def status_text() -> str:
    """Render the status block from config, the state file and a health probe."""
    package = _package()
    from importlib import import_module

    config_module = import_module(f"{package.__name__}.config")
    client_module = import_module(f"{package.__name__}.client")
    tools_module = import_module(f"{package.__name__}.tools")
    mcp_module = import_module(f"{package.__name__}.mcp_proxy")

    config = config_module.load_config()
    client = client_module.EngramClient(config)
    hermes_home = config_module.resolve_hermes_home()
    return tools_module.status_report(
        client,
        hermes_home,
        project=config.project_override,
        tools_label=mcp_module.recorded_label(hermes_home),
    )


def show_text(observation_id: str) -> str:
    """Render one observation's detail view."""
    package = _package()
    from importlib import import_module

    config_module = import_module(f"{package.__name__}.config")
    client_module = import_module(f"{package.__name__}.client")
    tools_module = import_module(f"{package.__name__}.tools")

    client = client_module.EngramClient(config_module.load_config())
    return tools_module.observation_report(client, observation_id)


def recall_command(args: Any) -> int:
    """Dispatch ``hermes recall <subcommand>``. Prints plain text; returns an exit code."""
    action = getattr(args, "recall_action", None)
    if action == "status" or action is None:
        print(status_text())
        return 0
    if action == "show":
        print(show_text(str(getattr(args, "id", "") or "")))
        return 0
    print("Usage: hermes recall <status|show ID>")
    return 2


def register_cli(subparser: Any) -> None:
    """Build the ``hermes recall`` argparse tree."""
    subs = subparser.add_subparsers(dest="recall_action")
    subs.add_parser("status", help="Show Engram recall status and the last memory id")
    show = subs.add_parser("show", help="Show one Engram observation")
    show.add_argument("id", help="Observation id, e.g. 761")
    subparser.set_defaults(func=recall_command)
