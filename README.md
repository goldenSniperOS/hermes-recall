# hermes-recall

Engram™-compatible memory provider for
[Hermes Agent](https://github.com/NousResearch/hermes-agent). It brings the
always-on memory behavior of
[gentle-engram](https://github.com/Gentleman-Programming/engram/tree/main/plugin/pi)
(the Engram memory package for Pi, by Alan Buscaglia / Gentleman Programming)
to Hermes, through the native Hermes `MemoryProvider` interface.
Independent community project; see [Not affiliated](#not-affiliated).

<div align="center">

<a href="https://github.com/Gentleman-Programming/gentle-ai">
  <img width="220" src="https://raw.githubusercontent.com/Gentleman-Programming/gentle-ai/main/docs/assets/brand/built-with-gentle-ai.png" alt="Built with Gentle-AI" />
</a>

</div>

## Why

When Engram is only connected to Hermes as an MCP server, the model decides
whether to search memory. It often starts working without any recall, and
every API call carries ~19 Engram tool schemas.

hermes-recall makes memory **deterministic**:

| What | When | Who decides |
|---|---|---|
| Recall (search + first-turn context) | Before every non-trivial turn | Hermes, not the model |
| `🧠 Recall · <project> — recalled N memories` | Every turn with hits (CLI, TUI, Telegram...) | Hermes |
| Prompt capture | Every turn | Hermes |
| Passive capture of tool results and answers | After every turn, in the background | Hermes |
| Subagent results capture | When a subagent finishes | Hermes |
| Compaction archive + recovery context | Before context compression | Hermes |
| Session start / end | Session boundaries | Hermes |
| `mem_save`, `mem_search`, ... | When the model wants them | The model |

It talks to the same local Engram server (`engram serve`) as every other
agent, so memories, projects and ids are shared: a memory saved from Hermes
shows up in Pi, OpenCode or Claude Code with the same `#id`, and the other
way around.

## Requirements

- Hermes Agent with memory-provider plugin support (tested with v0.21.0).
- The Engram binary, with its HTTP server running:

  ```bash
  brew install gentleman-programming/tap/engram
  engram serve        # default http://127.0.0.1:7437
  ```

  Tested with Engram 2.0.0.

## Install

Paste the prompt from [Install with an agent](docs/install-with-an-agent.md)
into a chat with your Hermes agent, or do it by hand:

```bash
hermes plugins install goldenSniperOS/hermes-recall
hermes config set memory.provider recall
```

The installer prints "Plugin installed but not enabled". You can ignore it:
memory providers are not listed in `plugins.enabled`. Hermes activates
exactly one, by name, through `memory.provider`.

Then restart Hermes (send `/restart` from a gateway chat, or start a new CLI
session) and check it:

```bash
hermes memory status     # Provider: recall ... available
hermes recall status     # project, Engram health, last memory id
```

Update later with `hermes plugins update recall`, then restart. Go back to
built-in memory only with `hermes memory off`.

### Engram MCP server

If you registered Engram as an MCP server before (`hermes mcp add engram ...`),
you can keep it, but hermes-recall already exposes the memory tools, so the
MCP tools are duplicates that cost tokens on every call. To remove them:

```bash
hermes mcp remove engram
```

## Usage

Nothing to do: recall and capture run on every turn. A turn with hits starts
with a line such as:

```text
🧠 Recall · my-project — recalled 3 memories
```

The model gets these tools (compact schemas, about 4 KB in total):

| Tool | Engram endpoint |
|---|---|
| `mem_search` | `GET /search` |
| `mem_save` | `POST /observations` |
| `mem_update` | `PATCH /observations/{id}` |
| `mem_get_observation` | `GET /observations/{id}` |
| `mem_context` | `GET /context` |
| `mem_session_summary` | `POST /observations` (type `session_summary`) |
| `mem_timeline` | `GET /timeline` |
| `mem_judge` | `POST /conflicts/judge` |
| `mem_pin`, `mem_unpin` | `PUT` / `DELETE /observations/{id}/pin` |

Every write result carries a `status` such as `🧠 saved #761`, and the last
memory id is kept in `<hermes_home>/recall/state.json`, so you can always get
back to it:

- `/recall` in any chat, or `hermes recall status` in a terminal: project,
  Engram health, last memory id and action, last recall count.
- `/recall <id>` or `hermes recall show <id>`: title, type, date and the first
  800 characters of that memory.

Subagents, cron jobs and memory flushes read memory but never write it.

### Limitations

- Hermes gives memory providers one native status line, the recall line.
  There is no supported way for a provider to print `🧠 saved #id` from a tool
  call, so that status lives in the tool result, the state file and `/recall`.
  See [docs/design.md](docs/design.md#findings).
- With Engram down, Hermes keeps working without recall or capture, and
  `hermes recall status` reports the server as unreachable.

## Configuration

All settings are optional:

| Setting | Default | Meaning |
|---|---|---|
| `memory.recall.url` | `http://127.0.0.1:7437` | Engram HTTP server. The `ENGRAM_URL` environment variable wins. |
| `memory.recall.project` | auto | Engram project. Empty means the project Engram detects for the working directory. |
| `memory.recall.recall_limit` | `5` | Search hits injected per turn (1-20). |
| `memory.recall.recall_max_chars` | `2500` | Cap on the injected recall block. |
| `memory.recall.enabled` | `true` | `false` turns recall, capture and tools off without uninstalling. |

```bash
hermes config set memory.recall.recall_limit 8
```

Hermes answers "not a recognized config key — it was saved anyway". That is
expected: provider settings are declared by the plugin, not by Hermes core,
and the provider reads them.

Design notes, the hook-to-endpoint map, threading, every cap and the stable
state file format are in [docs/design.md](docs/design.md).

## Credits and acknowledgements

This project exists because of the work of **Alan Buscaglia** and the
**Gentleman Programming** community. Engram, its memory protocol, its HTTP
API, and the always-on Pi behavior that hermes-recall reproduces
(gentle-engram) are theirs. hermes-recall only adapts that behavior to the
Hermes `MemoryProvider` hooks. Thank you for building them in the open and
under the MIT license.

hermes-recall itself is built with Gentle AI: planned and implemented
with the Organic Driven Development (ODD) workflow.

Thanks as well to Nous Research for Hermes Agent. hermes-recall uses its
public plugin API and copies none of its code.

Derived portions and upstream license texts are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Not affiliated

hermes-recall is an independent community project. It is **not** an
official Engram, Gentle AI, gentle-engram, or Hermes Agent project, and it is
not sponsored, endorsed, certified, or partnered with Gentleman Programming,
Alan Buscaglia, or Nous Research. Their names are used nominatively, only to
describe what this project is based on and compatible with.

Engram, Engram Cloud, Gentle AI, Gentle-AI and gentle-ai are trademarks of
Alan Buscaglia. Hermes Agent is a project of Nous Research. All other names
and trademarks belong to their respective owners.

## License

MIT. See [LICENSE](LICENSE).
