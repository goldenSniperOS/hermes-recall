# Install hermes-recall with an agent

You can install hermes-recall by hand by following the [Install](../README.md#install)
section of the README. You can also hand the job to the Hermes agent itself:
copy the prompt below and paste it into a chat with that agent (CLI, TUI or
any gateway such as Telegram).

The same prompt covers three starting points. The agent detects which one
applies to it:

| Starting point | What changes for you |
|---|---|
| **A. No Engram** (the `engram` binary is missing) | The agent installs Engram, installs the plugin and selects it as the memory provider. hermes-recall starts `engram serve` by itself on the next session. |
| **B. Engram already used by Hermes as an MCP server** (`hermes mcp list` shows `engram`) | Same as A without the binary install. The agent recommends removing the Engram MCP server and asks you first: hermes-recall now serves the same Engram MCP tools itself, bound to the Hermes session, so the separate server is redundant. Your other agents are not touched. |
| **C. Another memory provider is active** (`hermes memory status` shows a provider other than `recall`) | Hermes runs one external memory provider at a time. Selecting `recall` replaces the current one, so the agent asks you before switching. Nothing is deleted: you can switch back with `hermes config set memory.provider <old name>`. |

You can tell the starting points apart with:

```bash
engram --version
hermes mcp list
hermes memory status
```

## The prompt

````text
Install and configure the Hermes memory provider "hermes-recall"
(https://github.com/goldenSniperOS/hermes-recall) on this machine.
Run each step, and report the real output of each one. Never invent results.

HARD RULES
- Do not edit ~/.hermes/config.yaml by hand; use `hermes` CLI commands.
- Only touch the active Hermes profile.
- Never delete or reset memory data (no `hermes memory reset`, no edits to
  ~/.engram/). hermes-recall uses the existing Engram database as it is.
- Ask me before switching memory.provider if another provider is active,
  and before removing the Engram MCP server.
- If a command needs a password, administrator rights, or a credential,
  stop and ask me.

STEP 0 - Detect the starting point and tell me which one applies:
  - `engram --version` (missing = not installed)
  - `hermes mcp list` (is engram registered as an MCP server?)
  - `hermes memory status` (which provider is active?)
  - `hermes plugins list` (is recall already there?)
  A = no Engram. B = Engram registered as MCP for Hermes.
  C = another memory provider is active (can combine with B).

STEP 1 - Engram binary (the server is started automatically):
  If missing: brew install gentleman-programming/tap/engram
  If present: brew upgrade gentleman-programming/tap/engram
  If Homebrew is missing, stop and tell me.
  Check the binary: `engram instance-id` prints a 32-character id.
  You do not need to start the server: hermes-recall starts `engram serve`
  by itself when the new session begins (STEP 5) and nothing answers.
  Only if `hermes recall status` later shows "not running", start it by
  hand with `engram serve` in the background as a fallback (or ask me how I
  keep it running: launchd, systemd, a terminal) and check again.

STEP 2 - Plugin:
  New install:       hermes plugins install goldenSniperOS/hermes-recall
  Already installed: hermes plugins update recall
  Check: `hermes plugins list` shows recall.

STEP 3 - Select the provider (in starting point C, ask me first):
  hermes config set memory.provider recall
  Memory providers are not listed in plugins.enabled; do not run
  `hermes plugins enable` for it.

STEP 4 - Engram MCP server (starting point B only, ask me first):
  hermes-recall runs Engram's own MCP tools itself (mem_search, mem_save and
  the rest, from `engram mcp --tools=agent`) and binds every call to the
  Hermes session and project. A separate Engram MCP server in Hermes is
  therefore redundant: its tools are duplicates that cost tokens on every
  call and write to a session of the wrong directory. Recommend removing it.
  If I agree: hermes mcp remove engram

STEP 5 - Restart so the provider loads.
  If you run inside the gateway, you cannot restart yourself: tell me to send
  /restart in the chat. If you run in the CLI, tell me to start a new
  session. Then stop and wait.

STEP 6 - After the restart (I will message you again):
  - Run `hermes memory status` and show me the provider line.
  - Run `hermes recall status` and show me the output. Its "tools:" line
    should read "Engram MCP, N tools (Engram <version>)"; "HTTP fallback"
    means the Engram binary or the local default address is missing, and
    the line says why.
  - Ask me something that needs memory (for example "what did we do last
    time on this project?"). The reply should start with a line like
    "🧠 Recall · <project> — recalled N memories" when there are hits.

Finish with a short summary: starting point, Engram version and health,
plugin version, active memory provider, MCP change (if any), and anything
still pending.
````

## After the install

- Update later with `hermes plugins update recall`, then restart.
- `hermes recall status` (or `/recall` in a chat) shows the project, the
  Engram health, where the memory tools come from, the last memory id and the
  last recall count.
- `hermes config set memory.recall.tools '[mem_search, mem_save]'` limits the
  memory tools the model gets (all of them by default).
- To go back to built-in memory only: `hermes memory off`.
