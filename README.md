# OpenAI-compatible NVIDIA NIM and OpenRouter proxy

This proxy routes OpenAI-style chat completion requests to NVIDIA NIM or
OpenRouter and injects the same Frankenstein 5.4 roleplay preset before either
provider receives the conversation.

## Frankenstein 5.4 profile

`presets/frankenstein.json` is compiled from **Freaky Frankenstein 5.4
Internal States** and uses **FF5 Regex 3.0 Suite** with these choices on
Chub and other generic frontends:

- Cinematic Realism
- Third-person POV (replaces FF5's default Hybrid POV)
- Realism Mode
- Anti-Echo
- BOLT reasoning with a strict anti-drafting guard
- Micro NPC Voice 2.0
- Pop-in Graphics, Spectacle Combat, Onomatopoeia, and HQ NPC Genesis
- All eight Internal States modules: DnD Simulator, Internal Agenda, GM's
  Notebook, Inventory/Feats/Titles, Relationships, World Sim, Chekhov's Gun,
  and Internal Thoughts

The proxy expands FF5's SillyTavern `setvar`, `getvar`, `trim`, and `roll::1d20`
macros before sending the request upstream. On generic frontends, older
Internal State blocks are removed from model context after two turns while the
newest block is retained. Every exposed model routes through this Frankenstein
profile; legacy preset overrides are ignored.

The BOLT anti-drafting guard applies to private reasoning, not to the visible
scene. Before Internal States are generated, reasoning is limited to compact
decision notes (up to 220 words or 12 bullet lines), one selected next beat,
and turn-specific checks. It forbids sample narration, sample dialogue,
alternate scene drafts, prompt restatement, and prewriting the final response.
The normal 400–600-word visible response target is unchanged.

## Frontend URLs

Use the normal route for Chub AI and other generic OpenAI-compatible clients:

```text
https://YOUR-PROXY/v1/chat/completions
```

The generic route preserves FF5's inline-HTML Pop-in Graphics and Internal
States presentation and applies the included Regex 3.0 display suite. If a model
returns a Markdown or hidden-comment state variant, the proxy converts it into
one visible collapsible fallback panel so state display remains deterministic.

Use the Janitor-specific route in Janitor AI:

```text
https://YOUR-PROXY/janitor/v1/chat/completions
```

That route converts Pop-in Graphics and Internal States to portable Markdown.
The Janitor override generates the final Internal States record at the end of
the model's private reasoning, before the visible narrative. The normal
reasoning stream therefore places it inside the leading `<think>` box without
buffering or delaying the visible response. Use `SHOW_REASONING=true` on this
route so the think box and its state record remain in conversation history.
Regex 3.0's thought cleanup is applied only after the newest state has been
recovered for model context, so it cannot discard state stored inside that box.

## OpenRouter routes

For Chub and other generic OpenAI-compatible clients, use this API base:

```text
https://YOUR-PROXY/openrouter/v1
```

Its full chat endpoint is
`https://YOUR-PROXY/openrouter/v1/chat/completions`. It sends the compiled FF5
system prompt to OpenRouter, preserves OpenRouter-only request fields such as
`provider`, `models`, `plugins`, `tools`, and `reasoning`, and applies the same
generic FF5 response formatting used by the NIM route.

For Janitor AI, use this API base instead:

```text
https://YOUR-PROXY/janitor/openrouter/v1
```

Its full chat endpoint is
`https://YOUR-PROXY/janitor/openrouter/v1/chat/completions`. This keeps the
Janitor Markdown and `<think>` behavior while using OpenRouter for inference.
Both API bases also expose `/models` by forwarding OpenRouter's model list.

Send a canonical OpenRouter model ID such as `provider/model-name` in the
request. Canonical IDs pass through unchanged. If a frontend can only send an
alias such as `gpt-4`, set `OPENROUTER_MODEL` as a fallback or define explicit
aliases with `OPENROUTER_MODEL_MAPPING`.

## Environment

Set the key for each provider route you use:

```text
NIM_API_KEY=your_nvidia_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

Common optional settings:

```text
NIM_API_BASE=https://integrate.api.nvidia.com/v1
OPENROUTER_API_BASE=https://openrouter.ai/api/v1
OPENROUTER_MODEL=provider/model-name
OPENROUTER_MODEL_MAPPING={"gpt-4":"provider/model-name"}
OPENROUTER_SITE_URL=https://your-site.example
OPENROUTER_APP_NAME=FF5 Proxy
PORT=3000
REQUEST_TIMEOUT=600000
SHOW_REASONING=false
ENABLE_THINKING_MODE=false
REASONING_EFFORT=high
REASONING_BUDGET=16384
```

`OPENROUTER_MODEL` is only a fallback for non-canonical aliases; a request that
already contains a canonical `provider/model-name` still uses that model.
`OPENROUTER_MODEL_MAPPING` is an optional JSON object and takes priority over
the fallback. The site URL and app name are optional OpenRouter attribution
headers.

`ENABLE_THINKING_MODE` is the universal switch. The proxy translates it into
the request format used by each recognized model family instead of treating
every model like GLM:

| Model family | Upstream configuration |
| --- | --- |
| DeepSeek V4 | `thinking` plus `reasoning_effort` |
| GLM | `enable_thinking` |
| Kimi K3 | Top-level `reasoning_effort` |
| Earlier Kimi models | `thinking` |
| Qwen/QwQ | `enable_thinking` plus top-level `reasoning_effort` |
| Nemotron | `enable_thinking` plus optional `reasoning_budget` |
| Inkling | `reasoning_effort` (`none` when disabled) |
| Native reasoners such as DeepSeek R1 and MiniMax | Provider defaults; no invented flags |

`REASONING_EFFORT` accepts `none`, `low`, `medium`, `high`, `xhigh`, or
`max`. `REASONING_BUDGET` is optional; omit it to use the provider default.
Recognized per-request options in root `chat_template_kwargs`, SDK-style
`extra_body.chat_template_kwargs`, and top-level `reasoning_effort` or
`reasoning_budget` are normalized automatically.

Reasoning responses are normalized from `reasoning_content`, `reasoning`,
`thinking_content`, `thinking`, `analysis`, or structured
`reasoning_details`. When display is enabled, Janitor receives one portable
`<think>` block. Generic clients continue to receive only the final answer,
including when a provider emits literal `<think>`, `<thinking>`, `<reasoning>`,
or `<analysis>` tags.

Kimi K3 is handled specially for multi-turn history. If a frontend returns the
previous Janitor `<think>` display only as ordinary message content, the proxy
reconstructs that leading block into `assistant.reasoning_content` before
sending history back to NIM. If a client already preserves native
`reasoning_content`, that provider-native value is retained. K3 reasoning
metadata is also preserved on proxy responses for clients that can round-trip
unknown OpenAI-compatible fields. When `REASONING_EFFORT` is not explicitly
configured, K3 keeps its native `max` reasoning-effort default instead of the
proxy's generic `high` fallback.

On OpenRouter routes, a caller-supplied `reasoning` object is preserved. When
the request does not supply one, an explicitly configured
`ENABLE_THINKING_MODE` is translated to OpenRouter's unified `reasoning`
object. `REASONING_BUDGET` becomes `reasoning.max_tokens`; otherwise
`REASONING_EFFORT` becomes `reasoning.effort`. Generic routes ask OpenRouter not
to return reasoning text, while Janitor can receive it only when
`SHOW_REASONING=true`.

## Run and verify

```bash
npm install
npm test
npm start
```
