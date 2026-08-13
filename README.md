# OpenAI-compatible NVIDIA NIM proxy

This proxy routes OpenAI-style chat completion requests to NVIDIA NIM and
injects model-specific roleplay presets.

## Frankenstein 5.2 profile

`presets/frankenstein.json` is compiled from **FF5.2 Internal States Forced
Reasoning hapuppy** and uses **FF5 Regex Suite 2.4** with these choices on
Chub and other generic frontends:

- Cinematic Realism
- Third-person POV (replaces FF5's default Hybrid POV)
- Realism Mode
- Anti-Echo
- BOLT reasoning
- Micro NPC Voice
- Default Internal States: DnD Simulator, GM's Notebook, Relationships,
  Chekhov's Gun, and Internal Thoughts

The proxy expands FF5's SillyTavern `setvar`, `getvar`, `trim`, and `roll::1d20`
macros before sending the request to NVIDIA. On generic frontends, older
Internal State blocks are removed from model context after two turns while the
newest block is retained. Every exposed model routes through this Frankenstein
profile; legacy preset overrides are ignored.

## Frontend URLs

Use the normal route for Chub AI and other generic OpenAI-compatible clients:

```text
https://YOUR-PROXY/v1/chat/completions
```

The generic route preserves FF5's inline-HTML Pop-in Graphics and Internal
States presentation and applies the included FF5 display regex. If a model
returns a Markdown or hidden-comment state variant, the proxy converts it into
one visible collapsible fallback panel so state display remains deterministic.

Use the Janitor-specific route in Janitor AI:

```text
https://YOUR-PROXY/janitor/v1/chat/completions
```

That route converts Pop-in Graphics to portable Markdown and disables all nine
Internal State prompts (the eight modules plus the master output prompt). Any
state block left in older Janitor history is removed before the model request,
and any state tail emitted despite the instruction is dropped from streaming
and non-streaming responses. Janitor therefore receives narrative and Markdown
graphics only.

## Environment

Required:

```text
NIM_API_KEY=your_nvidia_api_key
```

Common optional settings:

```text
NIM_API_BASE=https://integrate.api.nvidia.com/v1
PORT=3000
REQUEST_TIMEOUT=600000
SHOW_REASONING=false
ENABLE_THINKING_MODE=false
```

## Run and verify

```bash
npm install
npm test
npm start
```
