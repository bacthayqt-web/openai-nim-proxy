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
    'gpt-4': 'z-ai/glm5',
    'gpt-4-turbo': 'deepseek-ai/deepseek-v3.1',
    'gpt-4o': 'deepseek-ai/deepseek-v3.2',
    'gpt-4-0613': 'minimaxai/minimax-m2.7',
    'claude-3-opus': 'moonshotai/kimi-k2-thinking',
    'claude-3-sonnet': 'z-ai/glm4.7',
    'gemini-pro': 'deepseek-ai/deepseek-v3.1-terminus'
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

    return existingSystemMsgs.concat(systemPresets).concat(nonSystemPresets).concat(nonSystemMsgs);
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
// Keywords that indicate a formatting instruction already exists
var FORMAT_KEYWORDS = [
'CRITICAL INSTRUCTION',
'double line breaks',
'paragraph structure',
'use markdown',
'formatting rule'
];

// Check if any system message already contains a formatting instruction
function hasFormattingKeyword(content) {
    if (!content || typeof content !== 'string') return false;
    var lower = content.toLowerCase();
    for (var i = 0; i < FORMAT_KEYWORDS.length; i++) {
        if (lower.indexOf(FORMAT_KEYWORDS[i].toLowerCase()) !== -1) {
            return true;
        }
    }
    return false;
}

var formattingNudge = {
    role: 'system',
    content: 'CRITICAL INSTRUCTION: Use Markdown. ALWAYS use double line breaks (\\n\\n) between paragraphs. No walls of text.'
};

var hasFormattingInstruction = messages.some(function(msg) {
    return msg.role === 'system' && hasFormattingKeyword(msg.content);
});

var enhanced;
if (hasFormattingInstruction) {
    // Only prepend the nudge to the FIRST matching system message, skip the rest
    var nudgeApplied = false;
    enhanced = messages.map(function(msg) {
        if (!nudgeApplied && msg.role === 'system' && hasFormattingKeyword(msg.content)) {
            nudgeApplied = true;
            return Object.assign({}, msg, {
                content: formattingNudge.content + '\n\n' + msg.content
            });
        }
        return msg;
    });
} else {
    enhanced = [formattingNudge].concat(messages);
}

// GLM models benefit from an extra formatting hint appended to the last user message
if (model.indexOf('glm') !== -1) {
    // Find the last user message (not just the last message)
    var lastUserIndex = -1;
    for (var i = enhanced.length - 1; i >= 0; i--) {
        if (enhanced[i].role === 'user') {
            lastUserIndex = i;
            break;
        }
    }
    if (lastUserIndex !== -1) {
        enhanced[lastUserIndex] = Object.assign({}, enhanced[lastUserIndex], {
            content: enhanced[lastUserIndex].content + '\n\n[Formatting Rule: Use clear, separate paragraphs with double line breaks.]'
        });
    }
}

return enhanced;

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
        var supportsThinking = nimModel.indexOf('deepseek') !== -1 || nimModel.indexOf('thinking') !== -1;

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
        }

        var response = await axios.post(
            NIM_API_BASE + '/chat/completions',
            nimRequest,
            {
                headers: {
                    Authorization: 'Bearer ' + NIM_API_KEY,
                    'Content-Type': 'application/json'
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
            handleStream(response.data, res);
        } else {
            handleNonStream(response.data, model, res);
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

function processDelta(delta) {
    if (!delta || !SHOW_REASONING) return delta;

    var reasoning = delta.reasoning_content;
    var content = delta.content;

    if (reasoning) {
        if (!processDelta._active) {
            delta.content = THINK_OPEN + '\n' + reasoning;
            processDelta._active = true;
        } else {
            delta.content = reasoning;
        }
        delete delta.reasoning_content;
    } else if (content && processDelta._active) {
        delta.content = '\n' + THINK_CLOSE + '\n\n' + content;
        processDelta._active = false;
    }

    return delta;
}

processDelta._active = false;

function handleStream(inputStream, res) {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');

    var buffer = '';
    var partialData = '';

    function safeWrite(obj) {
        try {
            var data = typeof obj === 'string' ? obj : JSON.stringify(obj);
            res.write('data: ' + data + '\n\n');
        } catch (e) {
            console.error('Stream write error:', e.message);
        }
    }

    function processChunk(rawData) {
        if (!rawData || rawData.trim() === '') return;
        if (rawData.trim() === '[DONE]') {
            safeWrite('[DONE]');
            return;
        }

        var parsed = null;
        try {
            parsed = JSON.parse(rawData);
        } catch (e) {
            partialData += rawData;
            try {
                parsed = JSON.parse(partialData);
                partialData = '';
            } catch (e2) {
                if (partialData.length > 100000) {
                    console.error('Partial data buffer exceeded, resetting');
                    partialData = '';
                }
                return;
            }
        }

        var delta = null;
        if (parsed && parsed.choices && parsed.choices[0]) {
            delta = parsed.choices[0].delta;
        }

        if (delta) {
            processDelta(delta);

            if (delta.content === null || delta.content === undefined || delta.content === '') {
                return;
            }
        }

        safeWrite(parsed);
    }

    inputStream.on('data', function(chunk) {
        buffer += chunk.toString('utf8');
        var lines = buffer.split(/\r?\n/);
        buffer = lines.pop() || '';

        for (var i = 0; i < lines.length; i++) {
            if (lines[i].indexOf('data: ') !== 0) continue;
            processChunk(lines[i].slice(6));
        }
    });

    inputStream.on('end', function() {
        if (buffer.indexOf('data: ') === 0) {
            processChunk(buffer.slice(6));
        }

        if (processDelta._active) {
            safeWrite({
                choices: [{ delta: { content: '\n' + THINK_CLOSE } }]
            });
            processDelta._active = false;
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

function handleNonStream(data, model, res) {
    try {
        var openaiResponse = {
            id: 'chatcmpl-' + Date.now(),
            object: 'chat.completion',
            created: Math.floor(Date.now() / 1000),
            model: model,
            choices: (data.choices || []).map(function(choice, index) {
                var fullContent = (choice && choice.message && choice.message.content) || '';

                if (SHOW_REASONING && choice && choice.message && choice.message.reasoning_content) {
                    fullContent = THINK_OPEN + '\n' + choice.message.reasoning_content + '\n' + THINK_CLOSE + '\n\n' + fullContent;
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
