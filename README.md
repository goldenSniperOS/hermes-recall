# hermes-recall

An Engram-compatible memory provider for Hermes Agent.
It makes a local Engram HTTP server the agent's memory: recall runs before every
turn and capture after it, without the model having to ask for either. Session
lifecycle and compaction recovery are handled by the provider, and a compact
`mem_*` tool set replaces the Engram MCP surface.

## Status

v0.1.0 is in development.

## Install

```bash
hermes plugins install goldenSniperOS/hermes-recall --enable
```

Then activate it in your Hermes `config.yaml`:

```yaml
memory:
  provider: recall
```

## Credits

Credits, badges and the full "not affiliated" notice are added in T7.
