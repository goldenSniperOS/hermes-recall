# Third-party notices

hermes-recall is MIT-licensed (see [LICENSE](LICENSE)). Parts of it are
adapted from the MIT-licensed project below. Its copyright and permission
notice is reproduced here, as the MIT License requires, for every derived
portion.

hermes-recall is an independent community project and is not affiliated with,
sponsored or endorsed by the upstream authors. Engram™ and Gentle AI™ are
trademarks of Alan Buscaglia. See the README for the full notice.

## Engram (including gentle-engram)

- Project: Engram by Alan Buscaglia / Gentleman Programming
- URL: https://github.com/Gentleman-Programming/engram
- Copyright: Copyright (c) 2026 Alan Buscaglia
- Pinned commit: a65917f3072ab4f26780bcf52427eba3963bef46
- License: MIT

Behavior and text adapted from Engram:

- The memory protocol in the provider system prompt block (adapted from
  `plugin/pi/index.ts` `MEMORY_INSTRUCTIONS`).
- Private-tag redaction (adapted from `plugin/pi/private-redaction.js`).
- Status wording such as `saved #<id>` (adapted from
  `plugin/pi/memory-tool-chrome.js`).
- The HTTP endpoint mapping of the memory tools (adapted from
  `plugin/pi/index.ts` `callMemoryTool`), against the server routes in
  `internal/server/server.go`.

```text
MIT License

Copyright (c) 2026 Alan Buscaglia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
