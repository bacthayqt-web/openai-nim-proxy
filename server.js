const express = require('express');
const cors = require('cors');
const axios = require('axios');
var fs = require('fs');
var path = require('path');
var app = express();
var PORT = process.env.PORT || 3000;

var NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
var NIM_API_KEY = process.env.NIM_API_KEY;
var SHOW_REASONING = process.env.SHOW_REASONING === 'true';
var ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
var REQUEST_TIMEOUT = parseInt(process.env.REQUEST_TIMEOUT || '600000', 10);
var MAX_TEMPERATURE = 2.0;
var MAX_MAX_TOKENS = 128000;

var PRESETS_DIR = path.join(__dirname, 'presets');

var THINK_OPEN = '\u003Cthink\u003E';
var THINK_CLOSE = '\u003C/think\u003E';

function loadPreset(presetName) {
    var filePath = path.join(PRESETS_DIR, presetName + '.json');
    try {
        var raw = fs.readFileSync(filePath, 'utf8');
        return JSON.parse(raw);
    } catch (err) {
        console.warn('Could not load preset "' + presetName + '": ' + err.message);
        return null;
    }
}

var PRESET_FRANKENSTEIN = loadPreset('frankenstein');
var PRESET_FRANKIMSTEIN = loadPreset('frankimstein');
var PRESET_FREAKYDEEPY = loadPreset('freakydeepy');
var FF5_REGEX = loadPreset('ff5-regex') || [];

var MODEL_MAPPING = {
    'gpt-3.5-turbo': 'moonshotai/kimi-k2.6',
    'gpt-4': 'z-ai/glm-5.2',
    'gpt-4-turbo': 'thinkingmachines/inkling',
    'gpt-4o': 'deepseek-ai/deepseek-v4-pro',
    'gpt-4-0613': 'deepseek-ai/deepseek-v4-flash',
    'claude-3-opus': 'google/gemma-4-31b-it',
    'claude-3-sonnet': 'nvidia/nemotron-3-ultra-550b-a55b',
    'gemini-pro': 'minimaxai/minimax-m3'
};

function isKimiModel(nimModelId) {
    if (!nimModelId) return false;
    var lower = nimModelId.toLowerCase();
    return lower.indexOf('moonshotai') !== -1 || lower.indexOf('kimi') !== -1;
}

function isDeepSeekModel(nimModelId) {
    if (!nimModelId) return false;
    return nimModelId.toLowerCase().indexOf('deepseek') !== -1;
}

function isInklingModel(nimModelId) {
    if (!nimModelId) return false;
    var lower = nimModelId.toLowerCase();
    return lower.indexOf('thinkingmachines') !== -1 || lower.indexOf('inkling') !== -1;
}

// Frontend is determined by which URL path the request came in on
// (Janitor AI's proxy field points at /janitor/v1/chat/completions).
function detectFrontend(req) {
    if (req.path.indexOf('/janitor/') === 0) return 'janitor';
    return 'default';
}

// SHOW_REASONING is intentionally scoped to the Janitor endpoint. Native
// thinking can stay enabled for every request, but generic clients receive
// only the final answer even when reasoning display is enabled globally.
function shouldShowReasoning(frontend) {
    return SHOW_REASONING && frontend === 'janitor';
}

function stripThinkBlocks(input) {
    var text = String(input || '');
    return text
        .replace(/<think\b[^>]*>[\s\S]*?<\/think\s*>/gi, '')
        .replace(/<think\b[^>]*>[\s\S]*$/gi, '')
        .replace(/^\s*<\/think\s*>/gi, '');
}

// Streaming equivalent of stripThinkBlocks. It handles markers split across
// arbitrary SSE/network chunk boundaries without buffering normal narrative.
function createThinkingStripStream() {
    var pending = '';
    var insideThinking = false;
    var openMarker = THINK_OPEN.toLowerCase();
    var closeMarker = THINK_CLOSE.toLowerCase();

    return {
        push: function(chunk) {
            pending += String(chunk || '');
            var visible = '';

            while (pending) {
                var lower = pending.toLowerCase();

                if (insideThinking) {
                    var closeAt = lower.indexOf(closeMarker);
                    if (closeAt === -1) {
                        pending = pending.slice(Math.max(0, pending.length - (closeMarker.length - 1)));
                        return visible;
                    }

                    pending = pending.slice(closeAt + closeMarker.length);
                    insideThinking = false;
                    continue;
                }

                var openAt = lower.indexOf(openMarker);
                if (openAt === -1) {
                    var safeLength = Math.max(0, pending.length - (openMarker.length - 1));
                    visible += pending.slice(0, safeLength);
                    pending = pending.slice(safeLength);
                    return visible;
                }

                visible += pending.slice(0, openAt);
                pending = pending.slice(openAt + openMarker.length);
                insideThinking = true;
            }

            return visible;
        },
        finish: function() {
            var remainder = insideThinking ? '' : pending;
            pending = '';
            insideThinking = false;
            return remainder;
        }
    };
}

// Per-frontend content overrides, keyed by prompt identifier. Unlike a full
// duplicated preset, this only swaps individual prompt entries (e.g. Janitor
// can't render raw inline HTML, so it gets a markdown-fenced version of just
// the immersive_graphics prompt) while everything else in the preset stays
// byte-for-byte identical across frontends.
var PROMPT_OVERRIDES = {
    janitor: loadPreset('overrides.janitor') || {}
};

// Janitor receives narrative and Markdown Pop-in Graphics only. Internal
// States are intentionally disabled there because Janitor has no dependable
// hidden metadata channel: HTML comments can surface in rendered messages.
// Chub and other generic clients keep the complete FF5 state stack.
var INTERNAL_STATE_PROMPT_IDS = [
    '019f62e8-892f-7021-97a6-42e1b83eaad3', // DnD Simulator
    '019f67b4-7381-7000-bcc4-496b2e6ed920', // Internal Agenda
    '019f67ad-c0b1-7000-aca4-0e2480fa02db', // GM Notebook
    '019f62e8-892f-7022-9eb9-e00c2944ebc6', // Inventory
    '019f62e8-892f-7023-825d-9351eca0347f', // Relationships
    '019f62e8-892f-7024-a40f-b906fceb58d2', // World Sim
    '019f62e8-892f-7025-be65-8859e7730ee0', // Chekhov's Gun
    '019f62e8-892f-7026-92ea-34ff510c244b', // Internal Thoughts
    '019f62e8-892f-7027-93ef-159f3d55c410'  // Internal States master
];

var PROMPT_EXCLUSIONS = {
    janitor: INTERNAL_STATE_PROMPT_IDS
};

function getPresetForModel(nimModelId) {
    return PRESET_FRANKENSTEIN;
}

function getOrderedPresetPrompts(preset) {
    if (!preset || !Array.isArray(preset.prompts)) return [];
    if (!Array.isArray(preset.prompt_order) || preset.prompt_order.length === 0) {
        return preset.prompts;
    }

    var byId = {};
    preset.prompts.forEach(function(prompt) {
        byId[prompt.identifier] = prompt;
    });

    return preset.prompt_order.map(function(identifier) {
        return byId[identifier];
    }).filter(Boolean);
}

// Expand the SillyTavern macros used inside FF5 while leaving ordinary card
// macros such as {{user}} and {{char}} untouched for the upstream frontend.
// Inner macros are evaluated first, which also supports getvar calls nested
// inside setvar templates.
function expandPresetMacros(input, variables) {
    var text = String(input || '');
    var vars = variables || {};
    var innermostMacro = /\{\{([^{}]*)\}\}/g;

    for (var pass = 0; pass < 1000; pass++) {
        var changed = false;
        text = text.replace(innermostMacro, function(full, body) {
            if (body.indexOf('//') === 0 || body.trim() === 'trim') {
                changed = true;
                return '';
            }

            var firstSeparator = body.indexOf('::');
            if (firstSeparator === -1) return full;

            var command = body.slice(0, firstSeparator).trim();
            var rest = body.slice(firstSeparator + 2);
            var secondSeparator = rest.indexOf('::');
            var name = secondSeparator === -1 ? rest : rest.slice(0, secondSeparator);
            var value = secondSeparator === -1 ? '' : rest.slice(secondSeparator + 2);

            if (command === 'setvar') {
                vars[name] = value;
                changed = true;
                return '';
            }
            if (command === 'getvar') {
                changed = true;
                return Object.prototype.hasOwnProperty.call(vars, name) ? vars[name] : value;
            }
            if (command === 'roll' && /^1d20$/i.test(name)) {
                changed = true;
                return String(Math.floor(Math.random() * 20) + 1);
            }

            return full;
        });

        // A setvar value may intentionally contain an ordinary frontend macro
        // such as {{user}}. The innermost-macro pass must preserve that macro,
        // but doing so also leaves its enclosing setvar unexpanded. Parse one
        // balanced setvar here so the template is stored without consuming the
        // ordinary card macro inside it.
        var setvarStart = text.indexOf('{{setvar::');
        if (setvarStart !== -1) {
            var macroDepth = 1;
            var macroCursor = setvarStart + 2;
            var setvarEnd = -1;
            while (macroCursor < text.length - 1) {
                var bracePair = text.slice(macroCursor, macroCursor + 2);
                if (bracePair === '{{') {
                    macroDepth += 1;
                    macroCursor += 2;
                    continue;
                }
                if (bracePair === '}}') {
                    macroDepth -= 1;
                    macroCursor += 2;
                    if (macroDepth === 0) {
                        setvarEnd = macroCursor;
                        break;
                    }
                    continue;
                }
                macroCursor += 1;
            }

            if (setvarEnd !== -1) {
                var setvarBody = text.slice(setvarStart + 2, setvarEnd - 2);
                var setvarPrefix = 'setvar::';
                var setvarRest = setvarBody.slice(setvarPrefix.length);
                var setvarSeparator = setvarRest.indexOf('::');
                if (setvarBody.indexOf(setvarPrefix) === 0 && setvarSeparator !== -1) {
                    var setvarName = setvarRest.slice(0, setvarSeparator);
                    var setvarValue = setvarRest.slice(setvarSeparator + 2);
                    vars[setvarName] = setvarValue;
                    text = text.slice(0, setvarStart) + text.slice(setvarEnd);
                    changed = true;
                }
            }
        }

        if (!changed) break;
    }

    return text.replace(/\n{3,}/g, '\n\n').trim();
}

function parseRegexLiteral(literal) {
    if (typeof literal !== 'string' || literal.charAt(0) !== '/') return null;
    var lastSlash = literal.lastIndexOf('/');
    if (lastSlash <= 0) return null;
    try {
        return new RegExp(literal.slice(1, lastSlash), literal.slice(lastSlash + 1));
    } catch (err) {
        console.warn('Invalid FF5 regex skipped: ' + err.message);
        return null;
    }
}

function runRegexScripts(text, scripts) {
    var output = String(text || '');
    (scripts || []).forEach(function(script) {
        if (!script || script.disabled) return;
        var regex = parseRegexLiteral(script.findRegex);
        if (regex) output = output.replace(regex, script.replaceString || '');
    });
    return output;
}

function prepareFF5History(messages, dropAllInternalStates) {
    var cleanupScripts = FF5_REGEX.filter(function(script) {
        // These are the machine-tag/context cleanup rules. Relationship-bar
        // styling is intentionally excluded from model context.
        return script.promptOnly && !script.markdownOnly;
    });

    var hiddenJanitorState = /<!--\s*FF5_INTERNAL_STATE\b[\s\S]*?END_FF5_INTERNAL_STATE\s*-->/g;

    return messages.map(function(message, index) {
        if (!message || typeof message.content !== 'string' || message.role !== 'assistant') {
            return message;
        }

        var depth = messages.length - 1 - index;
        var applicable = cleanupScripts.filter(function(script) {
            if (script.minDepth !== null && script.minDepth !== undefined && depth < script.minDepth) return false;
            if (script.maxDepth !== null && script.maxDepth !== undefined && depth > script.maxDepth) return false;
            return true;
        });

        var cleanedContent = runRegexScripts(message.content, applicable);
        // Janitor no longer uses Internal States, so purge every state variant
        // from old history. Generic clients retain the latest state for FF5
        // continuity and prune older copies after two turns.
        if (dropAllInternalStates) {
            cleanedContent = stripInternalState(cleanedContent);
        } else if (depth >= 2) {
            cleanedContent = cleanedContent.replace(hiddenJanitorState, '');
        }

        return Object.assign({}, message, { content: cleanedContent });
    });
}

function applyFrontendDisplay(text, frontend, enabled) {
    if (!enabled || frontend === 'janitor') return text;
    var displayScripts = FF5_REGEX.filter(function(script) {
        return !script.promptOnly;
    });
    return runRegexScripts(text, displayScripts);
}

// Detect the start of any known Internal States representation. This supports
// response cleanup for Janitor and deterministic display recovery for generic
// clients when a model deviates from the requested FF5 HTML template.
// Ordinary Pop-in Graphics are deliberately excluded.
function findInternalStateStart(input) {
    var text = String(input || '');
    var candidates = [];
    var patterns = [
        /<!--\s*FF5(?:[_\s-]*INTERNAL)?[_\s-]*STATES?\b/i,
        /<internal[_\s-]*states?\b/i,
        /<details\b[^>]*>\s*<summary\b[^>]*>[^<\n]{0,100}INTERNAL\s+STATES?\b/i,
        /(?:^|\n)[ \t]{0,3}(?:#{1,6}[ \t]+|\*\*|__)?(?:🎬[ \t]*)?INTERNAL\s+STATES?\b/im,
        /(?:^|\n)[ \t]*(?:\*\*)?\[(?:NPC AGENDAS|NPC LOCATIONS|FACTIONS|BONDS|QUESTS|INVENTORY(?:, FEATS & TITLES)?|CHEKHOV(?:'S)? GUN|INTERNAL THOUGHTS|GM(?:'S)? NOTEBOOK|DND TASK SIM|WORLD SIM|PHYSICS, ENGINE & WORLD)\](?:\*\*)?/im,
        /(?:^|\n)[ \t]{0,3}(?:#{1,6}[ \t]+|\*\*|__)?(?:GM(?:'S)? NOTEBOOK|DND TASK SIM|WORLD SIM|CHEKHOV(?:'S)? GUN|INTERNAL THOUGHTS|INVENTORY, FEATS & TITLES)\b/im
    ];

    patterns.forEach(function(pattern) {
        var match = pattern.exec(text);
        if (match) candidates.push(match.index);
    });

    if (candidates.length === 0) return -1;
    var start = Math.min.apply(Math, candidates);

    // Generic FF5 output wraps Internal States in GFX markers. Include the
    // opening wrapper in the hidden block, but do not consume unrelated phone,
    // terminal, letter, or map graphics.
    var gfxStart = text.lastIndexOf('<!-- GFX_START -->', start);
    if (gfxStart !== -1 && start - gfxStart <= 1024) {
        start = gfxStart;
    }

    return start;
}

function stripInternalState(input) {
    var text = String(input || '');
    var start = findInternalStateStart(text);
    if (start === -1) return text;

    var narrative = text.slice(0, start).replace(/[ \t]+$/g, '').replace(/\n{3,}$/g, '\n\n');
    return narrative.trimEnd();
}

function escapeHtml(input) {
    return String(input || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function normalizeGenericInternalState(input) {
    var state = String(input || '').trim();
    if (!state) return '';

    // Preserve the full native FF5 hierarchy when the model followed the
    // generic HTML template, adding wrappers only if they were omitted.
    if (/<internal[_\s-]*states?\b/i.test(state) && /<details\b/i.test(state)) {
        if (state.indexOf('<!-- GFX_START -->') === -1) {
            state = '<!-- GFX_START -->\n' + state;
        }
        if (state.indexOf('<!-- GFX_END -->') === -1) {
            state += '\n<!-- GFX_END -->';
        }
        return state;
    }

    // If a model emits Janitor-style comments or Markdown despite using the
    // generic route, convert it into one visible, collapsible fallback panel.
    var body = state
        .replace(/<!--\s*GFX_START\s*-->/gi, '')
        .replace(/<!--\s*GFX_END\s*-->/gi, '')
        .replace(/<!--\s*FF5(?:[_\s-]*INTERNAL)?[_\s-]*STATES?\b/gi, '')
        .replace(/END[_\s-]*FF5[_\s-]*INTERNAL[_\s-]*STATE\s*-->/gi, '')
        .replace(/<\/?internal[_\s-]*states?\b[^>]*>/gi, '')
        .replace(/(?:^|\n)[ \t]{0,3}(?:#{1,6}[ \t]+|\*\*|__)?(?:🎬[ \t]*)?INTERNAL\s+STATES?\b[^\n]*/i, '')
        .replace(/<!--|-->/g, '')
        .trim();

    if (!body) return '';

    return '<!-- GFX_START -->\n' +
        '<internal_states>\n' +
        '<details>\n' +
        '<summary>🎬 INTERNAL STATES</summary>\n' +
        '<pre style="white-space:pre-wrap;margin:0;">' + escapeHtml(body) + '</pre>\n' +
        '</details>\n' +
        '</internal_states>\n' +
        '<!-- GFX_END -->';
}

function displayGenericInternalState(input, frontend, enabled) {
    var text = String(input || '');
    var start = findInternalStateStart(text);
    if (start === -1) return applyFrontendDisplay(text, frontend, enabled);

    var narrative = text.slice(0, start).replace(/[ \t]+$/g, '').replace(/\n{3,}$/g, '\n\n');
    var state = normalizeGenericInternalState(text.slice(start));
    var combined = narrative.trimEnd() + (narrative.trim() && state ? '\n\n' : '') + state;
    return applyFrontendDisplay(combined, frontend, enabled);
}

// Compatibility alias retained for existing tests or imports. "Hide" now
// means remove from the response entirely; no state comment is returned.
function hideJanitorInternalState(input) {
    return stripInternalState(input);
}

function createInternalStateStream(frontend) {
    var pending = '';
    var stateBuffer = '';
    var stateStarted = false;
    // Retain only enough text to recognize a marker split across chunks. This
    // keeps ordinary narrative streaming with a small fixed delay instead of
    // buffering an entire short response.
    var lookbehind = 256;

    return {
        push: function(chunk) {
            var content = String(chunk || '');
            if (!content) return '';

            if (stateStarted) {
                stateBuffer += content;
                return '';
            }

            pending += content;
            var start = findInternalStateStart(pending);
            if (start !== -1) {
                var narrative = pending.slice(0, start);
                stateBuffer = pending.slice(start);
                pending = '';
                stateStarted = true;
                return narrative;
            }

            if (pending.length <= lookbehind) return '';
            var safeLength = pending.length - lookbehind;
            var safe = pending.slice(0, safeLength);
            pending = pending.slice(safeLength);
            return safe;
        },
        finish: function() {
            if (!stateStarted) {
                var remainder = pending;
                pending = '';
                return remainder;
            }

            var state = frontend === 'janitor'
                ? ''
                : normalizeGenericInternalState(stateBuffer);
            stateBuffer = '';
            stateStarted = false;
            pending = '';
            return state ? '\n\n' + state : '';
        },
        hasState: function() {
            return stateStarted;
        }
    };
}

function createJanitorStateStream() {
    return createInternalStateStream('janitor');
}

// FIX 1: Merge System Messages in Presets safely
function buildOrderedMessagesFromPreset(preset, originalMessages, promptOverrides, promptExclusions) {
    if (!preset || !preset.prompts || preset.prompts.length === 0) {
        return originalMessages;
    }

    var overrides = promptOverrides || {};
    var exclusions = promptExclusions || [];
    var internalStatesDisabled = exclusions.indexOf('019f62e8-892f-7027-93ef-159f3d55c410') !== -1;

    var macroVariables = {};
    var presetMessages = getOrderedPresetPrompts(preset)
        .filter(function(p) {
            return p.content && p.content.trim() !== '' && exclusions.indexOf(p.identifier) === -1;
        })
        .map(function(p) {
            var hasOverride = Object.prototype.hasOwnProperty.call(overrides, p.identifier);
            var content = hasOverride ? overrides[p.identifier] : p.content;
            if (internalStatesDisabled && p.identifier === 'jailbreak') {
                content = content.replace(
                    /Ensure internal states created correctly at end of every response\s*-\s*no need verbally to respond to this message\./i,
                    'Internal States are disabled on this frontend; never generate or append them.'
                );
            }
            return {
                role: p.role || 'system',
                content: expandPresetMacros(content, macroVariables)
            };
        })
        .filter(function(message) { return message.content !== ''; });

    var existingSystemMsgs = originalMessages.filter(function(m) { return m.role === 'system'; });
    var nonSystemMsgs = originalMessages.filter(function(m) { return m.role !== 'system'; });
    var systemPresets = presetMessages.filter(function(m) { return m.role === 'system'; });
    var nonSystemPresets = presetMessages.filter(function(m) { return m.role !== 'system'; });

    var allSystemMsgs = existingSystemMsgs.concat(systemPresets);
    var mergedSystemContent = allSystemMsgs.map(function(m) { return m.content; }).join('\n\n');

    var finalMessages = [];
    if (mergedSystemContent) {
        finalMessages.push({ role: 'system', content: mergedSystemContent });
    }

    return finalMessages.concat(nonSystemPresets).concat(nonSystemMsgs);
}

app.use(cors());
app.use(express.json({ limit: '5mb' }));
app.use(express.urlencoded({ limit: '5mb', extended: true }));

app.get('/v1/presets', function(req, res) {
    var presets = [];
    if (PRESET_FRANKENSTEIN) {
        presets.push({
            id: 'frankenstein',
            name: PRESET_FRANKENSTEIN.name,
            description: PRESET_FRANKENSTEIN.description,
            model_type: 'non-kimi'
        });
    }
    if (PRESET_FRANKIMSTEIN) {
        presets.push({
            id: 'frankimstein',
            name: PRESET_FRANKIMSTEIN.name,
            description: PRESET_FRANKIMSTEIN.description,
            model_type: 'kimi'
        });
    }
    if (PRESET_FREAKYDEEPY) {
        presets.push({
            id: 'freakydeepy',
            name: PRESET_FREAKYDEEPY.name,
            description: PRESET_FREAKYDEEPY.description,
            model_type: 'deepseek'
        });
    }
    res.json({ presets: presets });
});

function toBoolean(val) {
    return val === true || val === 'true';
}

function getEnhancedMessages(model, messages, allowHtmlUI) {
    var formattingNudge = {
        role: 'system',
        content: 'CRITICAL INSTRUCTION: Respond directly as text, never as JSON or a structured content array. Use blank lines between every narrative paragraph. Speech must use "double quotes"; actions and narration use *single asterisks*; emphasis uses **double asterisks**; thoughts use `backticks`.' +
            (allowHtmlUI
                ? '\n\nFF5 UI EXCEPTION: The Pop-in Graphics and Internal States blocks must use the raw inline HTML required by their own templates. Do not put those HTML blocks inside Markdown code fences.'
                : '\n\nJANITOR RENDERING: Use Markdown for the visible narrative and Pop-in Graphics. Internal States are disabled. Never generate an Internal States record, state heading, state XML, hidden state comment, or GFX-wrapped state panel. Never output visible raw HTML, CSS, details/summary tags, or GFX wrapper comments.')
    };

    var hasFormattingInstruction = messages.some(
        function(msg) {
            return msg.role === 'system' &&
                (msg.content.indexOf('Markdown') !== -1 ||
                 msg.content.indexOf('paragraph') !== -1 ||
                 msg.content.indexOf('formatting') !== -1 ||
                 msg.content.indexOf('CRITICAL INSTRUCTION') !== -1);
        }
    );

    var enhanced;
    if (hasFormattingInstruction) {
        enhanced = messages.map(function(msg) {
            if (msg.role === 'system' &&
                (msg.content.indexOf('Markdown') !== -1 ||
                 msg.content.indexOf('paragraph') !== -1 ||
                 msg.content.indexOf('formatting') !== -1)) {
                return Object.assign({}, msg, {
                    content: formattingNudge.content + '\n\n' + msg.content
                });
            }
            return msg;
        });
    } else {
        enhanced = [formattingNudge].concat(messages);
    }

    if (model.indexOf('glm') !== -1 || model.indexOf('deepseek') !== -1 ||
        model.indexOf('kimi') !== -1 || model.indexOf('moonshotai') !== -1) {
        var lastIndex = enhanced.length - 1;
        if (lastIndex >= 0 && enhanced[lastIndex].role === 'user') {
            enhanced[lastIndex] = Object.assign({}, enhanced[lastIndex], {
                content: enhanced[lastIndex].content + '\n\n[Formatting reminder: Every paragraph MUST be separated by a blank line (two newlines). Speech in "quotes", Actions in *asterisks*, Emphasis in **double asterisks**, Thoughts in `backticks`. Plain text only — no JSON.]'
            });
        }
    }

    return enhanced;
}

function cleanStructuredContent(text) {
    if (!text || typeof text !== 'string') {
        return text;
    }

    var trimmed = text.trim();

    if (trimmed.toLowerCase() === 'null') {
        return '';
    }

    var jsonParseAttempt = null;

    try {
        jsonParseAttempt = JSON.parse(trimmed);
    } catch (e1) {
        var fixed = trimmed.replace(/'/g, '"');
        try {
            jsonParseAttempt = JSON.parse(fixed);
        } catch (e2) {
        }
    }

    if (jsonParseAttempt === null) {
        return text;
    }

    if (Array.isArray(jsonParseAttempt)) {
        var resultParts = [];
        for (var i = 0; i < jsonParseAttempt.length; i++) {
            var item = jsonParseAttempt[i];
            if (item && typeof item === 'object') {
                if (item.type === 'text' && typeof item.text === 'string') {
                    resultParts.push(item.text);
                } else if (typeof item.text === 'string') {
                    resultParts.push(item.text);
                } else if (typeof item.content === 'string') {
                    resultParts.push(item.content);
                }
            } else if (typeof item === 'string') {
                resultParts.push(item);
            }
        }
        if (resultParts.length > 0) {
            return resultParts.join('\n');
        }
    }

    if (typeof jsonParseAttempt === 'object' && jsonParseAttempt !== null && !Array.isArray(jsonParseAttempt)) {
        if (jsonParseAttempt.type === 'text' && typeof jsonParseAttempt.text === 'string') {
            return jsonParseAttempt.text;
        }
        if (typeof jsonParseAttempt.text === 'string') {
            return jsonParseAttempt.text;
        }
        if (typeof jsonParseAttempt.content === 'string') {
            return jsonParseAttempt.content;
        }
    }

    return text;
}


function validateAndSanitizeParams(temperature, max_tokens) {
    var sanitizedTemp = temperature;
    if (temperature !== undefined && temperature !== null) {
        sanitizedTemp = Math.max(0, Math.min(MAX_TEMPERATURE, parseFloat(temperature)));
        if (isNaN(sanitizedTemp)) {
            sanitizedTemp = 0.7;
        }
    }

    var sanitizedMaxTokens = max_tokens;
    if (max_tokens !== undefined && max_tokens !== null) {
        sanitizedMaxTokens = Math.min(MAX_MAX_TOKENS, Math.max(1, parseInt(max_tokens, 10)));
        if (isNaN(sanitizedMaxTokens)) {
            sanitizedMaxTokens = 4096;
        }
    }

    var finalTemp = (sanitizedTemp !== undefined && sanitizedTemp !== null) ? sanitizedTemp : 0.7;
    var finalTokens = (sanitizedMaxTokens !== undefined && sanitizedMaxTokens !== null) ? sanitizedMaxTokens : 4096;
    return { temperature: finalTemp, max_tokens: finalTokens };
}

app.get('/health', function(req, res) {
    res.json({
        status: 'ok',
        reasoning_display: SHOW_REASONING,
        reasoning_display_scope: SHOW_REASONING ? 'janitor_only' : 'disabled',
        thinking_mode: ENABLE_THINKING_MODE,
        timeout_seconds: REQUEST_TIMEOUT / 1000,
        providers: {
            nim_configured: !!NIM_API_KEY
        },
        presets: {
            frankenstein: !!PRESET_FRANKENSTEIN,
            frankimstein: !!PRESET_FRANKIMSTEIN,
            freakydeepy: !!PRESET_FREAKYDEEPY
        }
    });
});

app.get('/v1/models', function(req, res) {
    var models = Object.keys(MODEL_MAPPING).map(function(id) {
        var nimModel = MODEL_MAPPING[id];
        var preset = getPresetForModel(nimModel);
        var presetLabel = 'none';
        if (preset) {
            var presetLower = preset.name.toLowerCase();
            if (presetLower.indexOf('kim') !== -1) {
                presetLabel = 'frankimstein';
            } else if (presetLower.indexOf('freaky') !== -1 || presetLower.indexOf('deepy') !== -1) {
                presetLabel = 'freakydeepy';
            } else {
                presetLabel = 'frankenstein';
            }
        }
        return {
            id: id,
            object: 'model',
            created: Math.floor(Date.now() / 1000),
            owned_by: 'nvidia-nim-proxy',
            nim_model: nimModel,
            preset: presetLabel
        };
    });
    res.json({ object: 'list', data: models });
});

app.post(['/v1/chat/completions', '/janitor/v1/chat/completions'], async function(req, res) {
    try {
        var model = req.body.model;
        var messages = req.body.messages;
        var temperature = req.body.temperature;
        var max_tokens = req.body.max_tokens;
        var stream = req.body.stream;
        var preset_override = req.body.preset_override;

        if (!messages || !Array.isArray(messages) || messages.length === 0) {
            return res.status(400).json({
                error: { message: 'Missing or invalid messages array', code: 400 }
            });
        }

        var sanitized = validateAndSanitizeParams(temperature, max_tokens);
        var wantsStream = toBoolean(stream);
        var nimModel = MODEL_MAPPING[model] || model;

        if (!NIM_API_KEY) {
            return res.status(500).json({
                error: {
                    message: 'NIM_API_KEY missing',
                    code: 500
                }
            });
        }

        // FIX 2 (extended): GLM caps for both tokens AND temperature
        if (nimModel.indexOf('glm') !== -1) {
            sanitized.max_tokens = Math.min(sanitized.max_tokens, 16384); // matches NVIDIA's GLM 5.2 reference (raised from 4096, which was a 5.1-era margin)
            sanitized.temperature = Math.min(sanitized.temperature, 1.0); // GLM max is 1.0
        }

        // Frankenstein is now the universal preset for every model. Ignore
        // legacy client overrides so no frontend can silently select an older
        // model-specific preset.
        var preset = getPresetForModel(nimModel);
        if (preset_override && preset_override !== 'frankenstein') {
            console.log('Preset override ignored: ' + preset_override + ' (universal Frankenstein routing is active)');
        }

        var frontend = detectFrontend(req);
        var processedMessages = messages;

        if (preset) {
            var promptOverrides = PROMPT_OVERRIDES[frontend];
            var promptExclusions = PROMPT_EXCLUSIONS[frontend] || [];
            var dropAllInternalStates = frontend === 'janitor';
            var sourceMessages = preset === PRESET_FRANKENSTEIN
                ? prepareFF5History(messages, dropAllInternalStates)
                : messages;
            processedMessages = buildOrderedMessagesFromPreset(
                preset,
                sourceMessages,
                promptOverrides,
                promptExclusions
            );
            console.log('Preset applied: ' + preset.name + ' for model ' + nimModel + ' (frontend: ' + frontend + ')');
            console.log('   - Preset prompts injected: ' + (preset.prompts.length - promptExclusions.length));
            if (dropAllInternalStates) {
                console.log('   - Internal States: DISABLED for Janitor');
            }
        } else {
            console.log('No preset available for model ' + nimModel + ', using raw messages');
        }

        var useFF5Display = preset === PRESET_FRANKENSTEIN;
        var allowHtmlUI = useFF5Display && frontend !== 'janitor';
        var enhancedMessages = getEnhancedMessages(nimModel, processedMessages, allowHtmlUI);

        // EXTRA SAFETY FIX: Guarantee only ONE system message ever exists for GLM compatibility
        var finalSystemMsgs = enhancedMessages.filter(function(m) { return m.role === 'system'; });
        var finalOtherMsgs = enhancedMessages.filter(function(m) { return m.role !== 'system'; });
        if (finalSystemMsgs.length > 1) {
            var combinedFinalSystem = finalSystemMsgs.map(function(m) { return m.content; }).join('\n\n');
            enhancedMessages = [{ role: 'system', content: combinedFinalSystem }].concat(finalOtherMsgs);
        }

        var supportsThinking = nimModel.indexOf('deepseek') !== -1
                           || nimModel.indexOf('thinking') !== -1
                           || nimModel.indexOf('glm') !== -1
                           || nimModel.indexOf('kimi') !== -1
                           || nimModel.indexOf('moonshotai') !== -1
                           || nimModel.indexOf('qwen') !== -1
                           || nimModel.indexOf('minimax') !== -1
                           || nimModel.indexOf('nemotron') !== -1;

        var nimRequest = {
            model: nimModel,
            messages: enhancedMessages,
            temperature: sanitized.temperature,
            max_tokens: sanitized.max_tokens,
            stream: wantsStream
        };

        // All chat_template_kwargs handling below is NIM-specific (it works around how NIM's
        // gateway merges vLLM/SGLang chat-template params).
        {
            if (isKimiModel(nimModel)) {
                // Kimi: chat_template_kwargs must be at ROOT payload level, not inside extra_body
                nimRequest.chat_template_kwargs = { thinking: ENABLE_THINKING_MODE };

            } else if (nimModel.indexOf('glm') !== -1) {
                // GLM: chat_template_kwargs must be at ROOT level (extra_body is an SDK abstraction,
                // not a real NIM API key — sending it as-is causes a 400). clear_thinking removed
                // as it caused a 400 on NIM for GLM 5.1; not re-added for 5.2 since that hasn't been
                // verified against NIM's endpoint specifically (other providers document it as valid
                // for preserved multi-turn thinking, but NIM's behavior may differ).
                nimRequest.chat_template_kwargs = {
                    enable_thinking: ENABLE_THINKING_MODE
                };

            } else if (nimModel.indexOf('deepseek') !== -1) {
                // DeepSeek: thinking OFF by default — enabling it causes extreme latency/timeouts.
                // Requires a separate ENABLE_DEEPSEEK_THINKING env flag to opt in explicitly.
                // chat_template_kwargs must be at ROOT payload level, not inside extra_body
                // (extra_body is an OpenAI-SDK-only abstraction; this server posts raw JSON via axios,
                // so a literal extra_body key is sent as-is and NIM will not merge it — same class of
                // bug that caused the GLM 400s).
                var deepseekThinking = process.env.ENABLE_DEEPSEEK_THINKING === 'true';
                nimRequest.chat_template_kwargs = { thinking: deepseekThinking };

            } else if (isInklingModel(nimModel)) {
                // Inkling reasons BY DEFAULT (per Thinking Machines' docs), so
                // ENABLE_THINKING_MODE=false must explicitly send reasoning_effort: "none" --
                // omitting the field does NOT disable it, unlike the other providers above.
                // Also note: Inkling takes a string enum here, not a boolean `thinking` flag,
                // and its model id ("thinkingmachines/inkling") would otherwise false-positive
                // match the generic `nimModel.indexOf('thinking')` check below, so this branch
                // must stay ahead of that catch-all.
                nimRequest.chat_template_kwargs = {
                    reasoning_effort: ENABLE_THINKING_MODE ? 'high' : 'none'
                };

            } else if (ENABLE_THINKING_MODE && supportsThinking) {
                // All other thinking-capable models (Qwen, Nemotron, MiniMax, etc.)
                // Same root-level requirement applies here.
                nimRequest.chat_template_kwargs = { thinking: true };
            }
        }

        var response = await axios.post(
            NIM_API_BASE + '/chat/completions',
            nimRequest,
            {
                headers: {
                    Authorization: 'Bearer ' + NIM_API_KEY,
                    'Content-Type': 'application/json',
                    // FIX 3: Force the gateway to stream properly
                    'Accept': wantsStream ? 'text/event-stream' : 'application/json'
                },
                responseType: wantsStream ? 'stream' : 'json',
                timeout: REQUEST_TIMEOUT,
                validateStatus: function() { return true; }
            }
        );

        if (response.status >= 400) {
            if (res.headersSent) return;

            // When wantsStream is true, responseType above was 'stream', so axios does NOT
            // parse the body even on error responses -- response.data is the raw Node stream
            // (sockets/agents included), not JSON. We have to read it ourselves to get NIM's
            // actual error text. On the non-stream path, response.data is already parsed JSON.
            var parsedErrorBody = null;
            if (wantsStream && response.data && typeof response.data.on === 'function') {
                var rawErrorBody = await new Promise(function(resolve) {
                    var chunks = [];
                    response.data.on('data', function(c) { chunks.push(c); });
                    response.data.on('end', function() { resolve(Buffer.concat(chunks).toString('utf8')); });
                    response.data.on('error', function() { resolve(''); });
                });
                try { parsedErrorBody = JSON.parse(rawErrorBody); } catch (e) { /* leave null */ }
                console.error('NVIDIA upstream ' + response.status + ' for model ' + nimModel + ' (raw body):', rawErrorBody);
            } else {
                parsedErrorBody = (response.data && typeof response.data === 'object') ? response.data : null;
                console.error('NVIDIA upstream ' + response.status + ' for model ' + nimModel + ':', JSON.stringify(parsedErrorBody));
            }

            var errorMessage = 'Upstream error';
            if (parsedErrorBody && parsedErrorBody.error) {
                // OpenAI-style nested shape: { error: { message, code } }
                errorMessage = parsedErrorBody.error.message || parsedErrorBody.error.code || errorMessage;
            } else if (parsedErrorBody && parsedErrorBody.message) {
                // vLLM/NIM flat shape: { object: "error", message, code }
                errorMessage = parsedErrorBody.message;
            }
            return res.status(response.status).json({
                error: { message: errorMessage, code: response.status }
            });
        }

        if (wantsStream) {
            handleStream(response.data, res, frontend, useFF5Display);
        } else {
            handleNonStream(response.data, model, res, frontend, useFF5Display);
        }
    } catch (error) {
        console.error('Proxy error:', {
            message: error.message,
            code: error.code,
            status: error.response ? error.response.status : undefined
        });
        if (!res.headersSent) {
            res.status(500).json({
                error: { message: error.message || 'Internal server error', code: 500 }
            });
        }
    }
});

function handleStream(inputStream, res, frontend, useFF5Display) {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');

    var buffer = '';
    var partialData = '';
    var reasoningActive = false;
    var exposeReasoning = shouldShowReasoning(frontend);
    var displayBuffer = '';
    var gfxStart = '<!-- GFX_START -->';
    var gfxEnd = '<!-- GFX_END -->';
    var internalStateStream = useFF5Display
        ? createInternalStateStream(frontend)
        : null;
    var thinkingStripStream = !exposeReasoning
        ? createThinkingStripStream()
        : null;

    function safeWrite(obj) {
        try {
            var data = typeof obj === 'string' ? obj : JSON.stringify(obj);
            res.write('data: ' + data + '\n\n');
        } catch (e) {
            console.error('Stream write error:', e.message);
        }
    }

    function processDelta(delta) {
        if (!delta) return;

        var isReasoningDelta = false;

        var reasoning = delta.reasoning_content;
        var content = delta.content;

        if (reasoning) {
            // Consume reasoning_content in both modes. Janitor receives it as
            // portable <think> text; every other frontend drops it completely.
            delete delta.reasoning_content;

            if (exposeReasoning) {
                isReasoningDelta = true;
                var cleanReasoning = cleanStructuredContent(reasoning);
                if (reasoningActive) {
                    delta.content = cleanReasoning;
                } else {
                    delta.content = '\u003Cthink\u003E\n' + cleanReasoning;
                    reasoningActive = true;
                }
            } else if (content) {
                delta.content = cleanStructuredContent(content);
            }
        } else if (content) {
            var cleanContent = cleanStructuredContent(content);
            if (exposeReasoning && reasoningActive) {
                delta.content = '\n\u003C/think\u003E\n\n' + cleanContent;
                reasoningActive = false;
            } else {
                delta.content = cleanContent;
            }
        } else if (!exposeReasoning && delta.reasoning_content !== undefined) {
            delete delta.reasoning_content;
        }

        // FIX 4: Allow the initial role chunk through to start UI sequence
        if (delta.role) {
            return true;
        }

        if (delta.content === null || delta.content === undefined) {
            return;
        }

        // Some providers may return literal <think> tags in content instead
        // of reasoning_content. Strip that fallback from every non-Janitor
        // response, including tags split across stream chunks.
        if (thinkingStripStream) {
            delta.content = thinkingStripStream.push(delta.content);
            if (delta.content === '') return;
        }

        // Hold a possible state tail until completion. Janitor drops it;
        // generic clients receive one normalized visible FF5 panel.
        if (internalStateStream) {
            // Native reasoning may discuss the phrase "Internal States" as
            // part of BOLT. Do not mistake displayed reasoning for the final
            // state record when SHOW_REASONING is intentionally enabled.
            if (isReasoningDelta) return true;
            delta.content = internalStateStream.push(delta.content);
            if (delta.content === '') return;
            if (frontend === 'janitor') return true;
        }

        if (!useFF5Display) return true;

        // Hold only complete FF5 GFX blocks. Narrative text continues streaming,
        // while a status/graphics block is released as soon as its closing marker
        // arrives and the FF5 display regex has been applied.
        displayBuffer += delta.content;
        var emitted = '';

        while (displayBuffer) {
            var startAt = displayBuffer.indexOf(gfxStart);
            if (startAt === -1) {
                var safeLength = Math.max(0, displayBuffer.length - (gfxStart.length - 1));
                emitted += displayBuffer.slice(0, safeLength);
                displayBuffer = displayBuffer.slice(safeLength);
                break;
            }

            emitted += displayBuffer.slice(0, startAt);
            displayBuffer = displayBuffer.slice(startAt);
            var endAt = displayBuffer.indexOf(gfxEnd);
            if (endAt === -1) break;

            var completeBlock = displayBuffer.slice(0, endAt + gfxEnd.length);
            emitted += applyFrontendDisplay(completeBlock, frontend, useFF5Display);
            displayBuffer = displayBuffer.slice(endAt + gfxEnd.length);
        }

        delta.content = emitted;
        if (delta.content === '') return;

        return true;
    }

    function processData(rawData) {
        if (!rawData || rawData.trim() === '') return;

        if (rawData.trim() === '[DONE]') {
            // Delay the terminal event until any buffered FF5 UI block has
            // been transformed and emitted.
            return;
        }

        try {
            var parsed = JSON.parse(rawData);
            var delta = null;
            if (parsed && parsed.choices && parsed.choices[0]) {
                delta = parsed.choices[0].delta;
            }

            if (delta) {
                var shouldSend = processDelta(delta);
                if (shouldSend) {
                    safeWrite(parsed);
                }
            }
        } catch (e) {
            partialData += rawData;
            try {
                var parsed2 = JSON.parse(partialData);
                partialData = '';

                var delta2 = null;
                if (parsed2 && parsed2.choices && parsed2.choices[0]) {
                    delta2 = parsed2.choices[0].delta;
                }
                if (delta2) {
                    var shouldSend2 = processDelta(delta2);
                    if (shouldSend2) {
                        safeWrite(parsed2);
                    }
                }
            } catch (e2) {
                if (partialData.length > 100000) {
                    console.error('Partial data buffer exceeded, resetting');
                    partialData = '';
                }
            }
        }
    }

    inputStream.on('data', function(chunk) {
        buffer += chunk.toString('utf8');
        var lines = buffer.split(/\r?\n/);
        buffer = lines.pop() || '';

        for (var i = 0; i < lines.length; i++) {
            if (lines[i].indexOf('data: ') !== 0) continue;
            var dataStr = lines[i].slice(6);
            processData(dataStr);
        }
    });

    inputStream.on('end', function() {
        if (buffer.indexOf('data: ') === 0) {
            processData(buffer.slice(6));
        }

        // Flush a short visible tail retained only to detect a split <think>
        // marker, then pass it through the existing state/display pipeline.
        if (thinkingStripStream) {
            var thinkRemainder = thinkingStripStream.finish();
            if (thinkRemainder) {
                if (internalStateStream) {
                    thinkRemainder = internalStateStream.push(thinkRemainder);
                }
                if (thinkRemainder) {
                    if (useFF5Display && frontend !== 'janitor') {
                        displayBuffer += thinkRemainder;
                    } else {
                        safeWrite({ choices: [{ delta: { content: thinkRemainder } }] });
                    }
                }
            }
        }

        if (internalStateStream) {
            var stateRemainder = internalStateStream.finish();
            if (stateRemainder) {
                if (frontend === 'janitor') {
                    safeWrite({
                        choices: [{ delta: { content: stateRemainder } }]
                    });
                } else {
                    displayBuffer += stateRemainder;
                }
            }
        }

        if (displayBuffer) {
            safeWrite({
                choices: [{ delta: { content: applyFrontendDisplay(displayBuffer, frontend, useFF5Display) } }]
            });
            displayBuffer = '';
        }

        if (exposeReasoning && reasoningActive) {
            safeWrite({
                choices: [{ delta: { content: '\n\u003C/think\u003E' } }]
            });
            reasoningActive = false;
        }

        safeWrite('[DONE]');
        res.end();
    });

    inputStream.on('error', function(err) {
        console.error('Stream error:', err.message);
        if (!res.headersSent) {
            res.status(500).json({
                error: { message: 'Stream processing error', code: 500 }
            });
        }
        res.end();
    });
}

function handleNonStream(data, model, res, frontend, useFF5Display) {
    try {
        var exposeReasoning = shouldShowReasoning(frontend);
        var openaiResponse = {
            id: 'chatcmpl-' + Date.now(),
            object: 'chat.completion',
            created: Math.floor(Date.now() / 1000),
            model: model,
            choices: (data.choices || []).map(function(choice, index) {
                var rawContent = (choice && choice.message && choice.message.content) || '';
                if (!exposeReasoning) {
                    rawContent = stripThinkBlocks(rawContent);
                }
                var cleanContent = cleanStructuredContent(rawContent);
                var fullContent = frontend === 'janitor' && useFF5Display
                    ? stripInternalState(cleanContent)
                    : displayGenericInternalState(cleanContent, frontend, useFF5Display);

                if (exposeReasoning && choice && choice.message && choice.message.reasoning_content) {
                    var rawReasoning = choice.message.reasoning_content;
                    var cleanReasoning = cleanStructuredContent(rawReasoning);
                    fullContent = '\u003Cthink\u003E\n' + cleanReasoning + '\n\u003C/think\u003E\n\n' + fullContent;
                }

                return {
                    index: choice.index !== undefined ? choice.index : index,
                    message: {
                        role: (choice && choice.message && choice.message.role) || 'assistant',
                        content: fullContent
                    },
                    finish_reason: choice.finish_reason || 'stop'
                };
            }),
            usage: data.usage || {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0
            }
        };

        res.json(openaiResponse);
    } catch (err) {
        console.error('Response formatting error:', err.message);
        res.status(500).json({
            error: { message: 'Response formatting error', code: 500 }
        });
    }
}

if (require.main === module) {
    app.listen(PORT, '0.0.0.0', function() {
        console.log('Proxy running on port ' + PORT);
        console.log('   - SHOW_REASONING: ' + SHOW_REASONING);
        console.log('   - ENABLE_THINKING_MODE: ' + ENABLE_THINKING_MODE);
        console.log('   - REQUEST_TIMEOUT: ' + (REQUEST_TIMEOUT / 1000) + 's');
        console.log('   - Frankenstein preset loaded: ' + (PRESET_FRANKENSTEIN ? 'YES' : 'NO'));
        console.log('   - FranKIMstein preset loaded: ' + (PRESET_FRANKIMSTEIN ? 'YES' : 'NO'));
        console.log('   - FreakyDeepy preset loaded: ' + (PRESET_FREAKYDEEPY ? 'YES' : 'NO'));

        if (!NIM_API_KEY) {
            console.warn('WARNING: NIM_API_KEY is missing!');
        }

        console.log('');
        console.log('Model -> Preset Mapping:');
        var modelKeys = Object.keys(MODEL_MAPPING);
        for (var i = 0; i < modelKeys.length; i++) {
            var openaiId = modelKeys[i];
            var nimId = MODEL_MAPPING[openaiId];
            var preset = getPresetForModel(nimId);
            var presetName = preset ? preset.name : 'NONE';
            var isKimi = isKimiModel(nimId) ? 'Kimi' : 'Non-Kimi';
            console.log('   - ' + openaiId + ' -> ' + nimId + ' (' + isKimi + ') -> ' + presetName);
        }
    });
}

// Export the Express app for serverless runtimes and expose pure helpers only
// under _test so regression tests exercise the exact production code.
module.exports = app;
module.exports._test = {
    detectFrontend: detectFrontend,
    shouldShowReasoning: shouldShowReasoning,
    stripThinkBlocks: stripThinkBlocks,
    createThinkingStripStream: createThinkingStripStream,
    findInternalStateStart: findInternalStateStart,
    stripInternalState: stripInternalState,
    normalizeGenericInternalState: normalizeGenericInternalState,
    displayGenericInternalState: displayGenericInternalState,
    hideJanitorInternalState: hideJanitorInternalState,
    createJanitorStateStream: createJanitorStateStream,
    createInternalStateStream: createInternalStateStream,
    handleStream: handleStream,
    handleNonStream: handleNonStream,
    prepareFF5History: prepareFF5History,
    buildOrderedMessagesFromPreset: buildOrderedMessagesFromPreset,
    expandPresetMacros: expandPresetMacros,
    getOrderedPresetPrompts: getOrderedPresetPrompts,
    internalStatePromptIds: INTERNAL_STATE_PROMPT_IDS
};
