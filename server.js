const express = require('express');
const cors = require('cors');
const axios = require('axios');
var fs = require('fs');
var path = require('path');
var app = express();
var PORT = process.env.PORT || 3000;

var NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
var NIM_API_KEY = process.env.NIM_API_KEY;
var SHOW_REASONING = process.env.SHOW_REASONING !== 'false';
var ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE !== 'false';
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

var MODEL_MAPPING = {
    'gpt-3.5-turbo': 'moonshotai/kimi-k2.5',
    'gpt-4': 'z-ai/glm-5.1',
    'gpt-4-turbo': 'deepseek-ai/deepseek-v3.1-terminus',
    'gpt-4o': 'deepseek-ai/deepseek-v3.2',
    'gpt-4-0613': 'minimaxai/minimax-m2.7',
    'claude-3-opus': 'moonshotai/kimi-k2-thinking',
    'claude-3-sonnet': 'z-ai/glm4.7',
    'gemini-pro': 'deepseek-ai/deepseek-v4-pro'
};

function isKimiModel(nimModelId) {
    if (!nimModelId) return false;
    var lower = nimModelId.toLowerCase();
    return lower.indexOf('moonshotai') !== -1 || lower.indexOf('kimi') !== -1;
}

function getPresetForModel(nimModelId) {
    if (isKimiModel(nimModelId)) {
        return PRESET_FRANKIMSTEIN;
    }
    return PRESET_FRANKENSTEIN;
}

// FIX 1: Merge System Messages in Presets safely
function buildOrderedMessagesFromPreset(preset, originalMessages) {
    if (!preset || !preset.prompts || preset.prompts.length === 0) {
        return originalMessages;
    }

    var presetMessages = preset.prompts
        .filter(function(p) { return p.content && p.content.trim() !== ''; })
        .map(function(p) {
            return {
                role: p.role || 'system',
                content: p.content.trim()
            };
        });

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
    res.json({ presets: presets });
});

function toBoolean(val) {
    return val === true || val === 'true';
}

function getEnhancedMessages(model, messages) {
    var formattingNudge = {
        role: 'system',
        content: 'CRITICAL INSTRUCTION: Use Markdown. You MUST respond with plain text only. Do NOT wrap your response in JSON, arrays, or structured formats like [{"type": "text", "text": "..."}]. Just write your response directly as plain text.\n\nSTRICT FORMATTING RULES:\n1. Paragraph Breaks: ALWAYS insert a blank line between paragraphs — that means two newline characters (one empty line) separating every paragraph, every time. Never run paragraphs together. No walls of text.\n2. Speech: Must ALWAYS be enclosed in "double quotes".\n3. Actions & Narration: Must ALWAYS be enclosed in *single asterisks*.\n4. Emphasis: Must ALWAYS be enclosed in **double asterisks**.\n5. Thoughts: Must ALWAYS be enclosed in `backticks`.'
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

    if (trimmed === 'null') {
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

// Returns true for models whose thinking should be hidden from the frontend.
function isThinkingHidden(nimModelId) {
    return nimModelId === 'z-ai/glm-5.1';
}

// Removes every <think>...</think> block and its contents from a string,
// leaving only the plain response text.
function stripThinkBlock(text) {
    if (!text || typeof text !== 'string') return text;
    var result = text;
    var openIdx = result.indexOf(THINK_OPEN);
    while (openIdx !== -1) {
        var closeIdx = result.indexOf(THINK_CLOSE, openIdx);
        if (closeIdx === -1) {
            result = result.slice(0, openIdx).trim();
            break;
        }
        result = result.slice(0, openIdx) + result.slice(closeIdx + THINK_CLOSE.length);
        openIdx = result.indexOf(THINK_OPEN);
    }
    return result.trim();
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
        thinking_mode: ENABLE_THINKING_MODE,
        timeout_seconds: REQUEST_TIMEOUT / 1000,
        presets: {
            frankenstein: !!PRESET_FRANKENSTEIN,
            frankimstein: !!PRESET_FRANKIMSTEIN
        }
    });
});

app.get('/v1/models', function(req, res) {
    var models = Object.keys(MODEL_MAPPING).map(function(id) {
        var nimModel = MODEL_MAPPING[id];
        var preset = getPresetForModel(nimModel);
        var presetLabel = 'none';
        if (preset) {
            presetLabel = preset.name.toLowerCase().indexOf('kim') !== -1 ? 'frankimstein' : 'frankenstein';
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

app.post('/v1/chat/completions', async function(req, res) {
    try {
        if (!NIM_API_KEY) {
            return res.status(500).json({
                error: { message: 'NIM_API_KEY missing', code: 500 }
            });
        }

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

        // FIX 2 (extended): GLM caps for both tokens AND temperature
if (nimModel.indexOf('glm') !== -1) {
    sanitized.max_tokens = Math.min(sanitized.max_tokens, 4096); // safer than 8192
    sanitized.temperature = Math.min(sanitized.temperature, 1.0); // GLM max is 1.0
}

        var preset;
        if (preset_override && (preset_override === 'frankenstein' || preset_override === 'frankimstein')) {
            preset = preset_override === 'frankimstein' ? PRESET_FRANKIMSTEIN : PRESET_FRANKENSTEIN;
            console.log('Preset override: ' + preset_override + ' (forced by client)');
        } else {
            preset = getPresetForModel(nimModel);
        }

        var processedMessages = messages;

        if (preset) {
            processedMessages = buildOrderedMessagesFromPreset(preset, messages);
            console.log('Preset applied: ' + preset.name + ' for model ' + nimModel);
            console.log('   - Preset prompts injected: ' + preset.prompts.length);
        } else {
            console.log('No preset available for model ' + nimModel + ', using raw messages');
        }

        var enhancedMessages = getEnhancedMessages(nimModel, processedMessages);

        // EXTRA SAFETY FIX: Guarantee only ONE system message ever exists for GLM compatibility
        var finalSystemMsgs = enhancedMessages.filter(function(m) { return m.role === 'system'; });
        var finalOtherMsgs = enhancedMessages.filter(function(m) { return m.role !== 'system'; });
        if (finalSystemMsgs.length > 1) {
            var combinedFinalSystem = finalSystemMsgs.map(function(m) { return m.content; }).join('\n\n');
            enhancedMessages = [{ role: 'system', content: combinedFinalSystem }].concat(finalOtherMsgs);
        }

        var supportsThinking = nimModel.indexOf('deepseek-r') !== -1  // R1, R2 variants
                           || nimModel.indexOf('thinking') !== -1
                           || nimModel.indexOf('glm') !== -1;        // GLM-4.7, GLM-5 think by default

        var nimRequest = {
            model: nimModel,
            messages: enhancedMessages,
            temperature: sanitized.temperature,
            max_tokens: sanitized.max_tokens,
            stream: wantsStream
        };

        if (ENABLE_THINKING_MODE && supportsThinking) {
            nimRequest.extra_body = {
                chat_template_kwargs: { thinking: true }
            };
        } else if (nimModel.indexOf('glm') !== -1) {
            // GLM thinks by default — explicitly disable if thinking mode is off
            nimRequest.extra_body = {
                chat_template_kwargs: { thinking: false }
            };
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
            var errorMessage = 'Upstream error';
            if (response.data && response.data.error) {
                errorMessage = response.data.error.message || response.data.error.code || errorMessage;
            }
            return res.status(response.status).json({
                error: { message: errorMessage, code: response.status }
            });
        }

        if (wantsStream) {
            handleStream(response.data, res, nimModel);
        } else {
            handleNonStream(response.data, model, res, nimModel);
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

function handleStream(inputStream, res, nimModel) {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');

    var buffer = '';
    var partialData = '';
    var reasoningActive = false;
    var hideThinking = isThinkingHidden(nimModel);

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

        if (hideThinking) {
            // GLM 5.1: consume reasoning silently, forward only actual content.
            // GLM can send reasoning_content and content in the SAME delta chunk
            // (the transition chunk). Must NOT null out content in that case or
            // the first characters of the response get dropped.
            if (delta.reasoning_content) {
                delete delta.reasoning_content;
                if (delta.content) {
                    // Transition chunk — reasoning ends and content starts together.
                    delta.content = cleanStructuredContent(delta.content);
                    reasoningActive = false;
                } else {
                    // Pure reasoning chunk — suppress entirely.
                    delta.content = null;
                    reasoningActive = true;
                    return;
                }
            } else if (delta.content && reasoningActive) {
                // First standalone content chunk after reasoning.
                delta.content = cleanStructuredContent(delta.content);
                reasoningActive = false;
            } else if (delta.content) {
                delta.content = cleanStructuredContent(delta.content);
            }
        } else if (SHOW_REASONING) {
            var reasoning = delta.reasoning_content;
            var content = delta.content;

            if (reasoning) {
                var cleanReasoning = cleanStructuredContent(reasoning);
                if (reasoningActive) {
                    delta.content = cleanReasoning;
                } else {
                    delta.content = '\u003Cthink\u003E\n' + cleanReasoning;
                    reasoningActive = true;
                }
                delete delta.reasoning_content;
            } else if (content) {
                var cleanContent = cleanStructuredContent(content);
                if (reasoningActive) {
                    delta.content = '\n\u003C/think\u003E\n\n' + cleanContent;
                    reasoningActive = false;
                } else {
                    delta.content = cleanContent;
                }
            }
        }

        // FIX 4: Allow the initial role chunk through to start UI sequence
        if (delta.role) {
            return true;
        }

        if (delta.content === null || delta.content === undefined) {
            return;
        }

        // Removed the strict `.trim() === ''` check here to allow spaces/newlines

        return true;
    }

    function processData(rawData) {
        if (!rawData || rawData.trim() === '') return;

        if (rawData.trim() === '[DONE]') {
            safeWrite('[DONE]');
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

        if (reasoningActive) {
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

function handleNonStream(data, model, res, nimModel) {
    try {
        var openaiResponse = {
            id: 'chatcmpl-' + Date.now(),
            object: 'chat.completion',
            created: Math.floor(Date.now() / 1000),
            model: model,
            choices: (data.choices || []).map(function(choice, index) {
                var rawContent = (choice && choice.message && choice.message.content) || '';
                var fullContent = cleanStructuredContent(rawContent);

                if (SHOW_REASONING && choice && choice.message && choice.message.reasoning_content) {
                    var rawReasoning = choice.message.reasoning_content;
                    var cleanReasoning = cleanStructuredContent(rawReasoning);
                    fullContent = '\u003Cthink\u003E\n' + cleanReasoning + '\n\u003C/think\u003E\n\n' + fullContent;
                }

                // For models with hidden thinking, strip the think block before sending
                if (isThinkingHidden(nimModel)) {
                    fullContent = stripThinkBlock(fullContent);
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

app.listen(PORT, '0.0.0.0', function() {
    console.log('Proxy running on port ' + PORT);
    console.log('   - SHOW_REASONING: ' + SHOW_REASONING);
    console.log('   - ENABLE_THINKING_MODE: ' + ENABLE_THINKING_MODE);
    console.log('   - REQUEST_TIMEOUT: ' + (REQUEST_TIMEOUT / 1000) + 's');
    console.log('   - Frankenstein preset loaded: ' + (PRESET_FRANKENSTEIN ? 'YES' : 'NO'));
    console.log('   - FranKIMstein preset loaded: ' + (PRESET_FRANKIMSTEIN ? 'YES' : 'NO'));

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
