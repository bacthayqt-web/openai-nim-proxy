const express = require('express');
const cors = require('cors');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const app = express();
const PORT = process.env.PORT || 3000;

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const SHOW_REASONING = process.env.SHOW_REASONING !== 'false';
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE !== 'false';
const REQUEST_TIMEOUT = parseInt(process.env.REQUEST_TIMEOUT || '600000', 10);
const MAX_TEMPERATURE = 2.0;
const MAX_MAX_TOKENS = 128000;

const PRESETS_DIR = path.join(__dirname, 'presets');

function loadPreset(presetName) {
const filePath = path.join(PRESETS_DIR, ${presetName}.json);
try {
const raw = fs.readFileSync(filePath, 'utf8');
return JSON.parse(raw);
} catch (err) {
console.warn('Could not load preset "' + presetName + '": ' + err.message);
return null;
}
}

const PRESET_FRANKENSTEIN = loadPreset('frankenstein');
const PRESET_FRANKIMSTEIN = loadPreset('frankimstein');

const MODEL_MAPPING = {
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
const lower = nimModelId.toLowerCase();
return lower.includes('moonshotai') || lower.includes('kimi');
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

const presetMessages = preset.prompts
    .filter(function(p) { return p.content && p.content.trim() !== ''; })
    .map(function(p) {
        return {
            role: p.role || 'system',
            content: p.content.trim()
        };
    });

const existingSystemMsgs = originalMessages.filter(function(m) { return m.role === 'system'; });
const nonSystemMsgs = originalMessages.filter(function(m) { return m.role !== 'system'; });

return [
    ...existingSystemMsgs,
    ...presetMessages.filter(function(m) { return m.role === 'system'; }),
    ...presetMessages.filter(function(m) { return m.role !== 'system'; }),
    ...nonSystemMsgs
];

}

app.use(cors());
app.use(express.json({ limit: '5mb' }));
app.use(express.urlencoded({ limit: '5mb', extended: true }));

app.get('/v1/presets', function(req, res) {
const presets = [];
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

const toBoolean = function(val) { return val === true || val === 'true'; };

const getEnhancedMessages = function(model, messages) {
const formattingNudge = {
role: 'system',
content: 'CRITICAL INSTRUCTION: Use Markdown. ALWAYS use double line breaks (\n\n) between paragraphs. No walls of text.'
};

const hasFormattingInstruction = messages.some(
    function(msg) {
        return msg.role === 'system' &&
            (msg.content.includes('Markdown') ||
             msg.content.includes('paragraph') ||
             msg.content.includes('formatting') ||
             msg.content.includes('CRITICAL INSTRUCTION'));
    }
);

let enhanced;
if (hasFormattingInstruction) {
    enhanced = messages.map(function(msg) {
        if (msg.role === 'system' &&
            (msg.content.includes('Markdown') ||
             msg.content.includes('paragraph') ||
             msg.content.includes('formatting'))) {
            return Object.assign({}, msg, {
                content: formattingNudge.content + '\n\n' + msg.content
            });
        }
        return msg;
    });
} else {
    enhanced = [formattingNudge].concat(messages);
}

if (model.includes('glm')) {
    const lastIndex = enhanced.length - 1;
    if (lastIndex >= 0 && enhanced[lastIndex].role === 'user') {
        enhanced[lastIndex] = Object.assign({}, enhanced[lastIndex], {
            content: enhanced[lastIndex].content + '\n\n[Formatting Rule: Use clear, separate paragraphs with double line breaks.]'
        });
    }
}

return enhanced;

};

const validateAndSanitizeParams = function(temperature, max_tokens) {
let sanitizedTemp = temperature;
if (temperature !== undefined && temperature !== null) {
sanitizedTemp = Math.max(0, Math.min(MAX_TEMPERATURE, parseFloat(temperature)));
if (isNaN(sanitizedTemp)) {
sanitizedTemp = 0.7;
}
}

let sanitizedMaxTokens = max_tokens;
if (max_tokens !== undefined && max_tokens !== null) {
    sanitizedMaxTokens = Math.min(MAX_MAX_TOKENS, Math.max(1, parseInt(max_tokens, 10)));
    if (isNaN(sanitizedMaxTokens)) {
        sanitizedMaxTokens = 4096;
    }
}

return {
    temperature: sanitizedTemp !== undefined && sanitizedTemp !== null ? sanitizedTemp : 0.7,
    max_tokens: sanitizedMaxTokens !== undefined && sanitizedMaxTokens !== null ? sanitizedMaxTokens : 4096
};

};

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
const models = Object.keys(MODEL_MAPPING).map(function(id) {
const nimModel = MODEL_MAPPING[id];
const preset = getPresetForModel(nimModel);
return {
id: id,
object: 'model',
created: Math.floor(Date.now() / 1000),
owned_by: 'nvidia-nim-proxy',
nim_model: nimModel,
preset: preset ? (preset.name.toLowerCase().includes('kim') ? 'frankimstein' : 'frankenstein') : 'none'
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

    const model = req.body.model;
    const messages = req.body.messages;
    const temperature = req.body.temperature;
    const max_tokens = req.body.max_tokens;
    const stream = req.body.stream;
    const preset_override = req.body.preset_override;

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
        return res.status(400).json({
            error: { message: 'Missing or invalid messages array', code: 400 }
        });
    }

    const sanitized = validateAndSanitizeParams(temperature, max_tokens);
    const sanitizedTemp = sanitized.temperature;
    const sanitizedMaxTokens = sanitized.max_tokens;

    const wantsStream = toBoolean(stream);
    const nimModel = MODEL_MAPPING[model] || model;

    let preset;
    if (preset_override && (preset_override === 'frankenstein' || preset_override === 'frankimstein')) {
        preset = preset_override === 'frankimstein' ? PRESET_FRANKIMSTEIN : PRESET_FRANKENSTEIN;
        console.log('Preset override: ' + preset_override + ' (forced by client)');
    } else {
        preset = getPresetForModel(nimModel);
    }

    let processedMessages = messages;

    if (preset) {
        processedMessages = buildOrderedMessagesFromPreset(preset, messages);
        console.log('Preset applied: ' + preset.name + ' for model ' + nimModel);
        console.log('   - Preset prompts injected: ' + preset.prompts.length);
    } else {
        console.log('No preset available for model ' + nimModel + ', using raw messages');
    }

    const enhancedMessages = getEnhancedMessages(nimModel, processedMessages);
    const supportsThinking = nimModel.includes('deepseek') || nimModel.includes('thinking');

    const nimRequest = {
        model: nimModel,
        messages: enhancedMessages,
        temperature: sanitizedTemp,
        max_tokens: sanitizedMaxTokens,
        stream: wantsStream
    };

    if (ENABLE_THINKING_MODE && supportsThinking) {
        nimRequest.extra_body = {
            chat_template_kwargs: { thinking: true }
        };
    }

    const response = await axios.post(
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

        const errorMessage = (response.data && response.data.error && response.data.error.message) ||
            (response.data && response.data.error && response.data.error.code) ||
            'Upstream error';

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
        status: error.response ? error.response.status : undefined,
        data: error.response ? error.response.data : undefined
    });

    if (!res.headersSent) {
        res.status(500).json({
            error: { message: error.message || 'Internal server error', code: 500 }
        });
    }
}

});

function handleStream(inputStream, res) {
res.setHeader('Content-Type', 'text/event-stream');
res.setHeader('Cache-Control', 'no-cache');
res.setHeader('Connection', 'keep-alive');
res.setHeader('X-Accel-Buffering', 'no');

let buffer = '';
let partialData = '';
let reasoningActive = false;

const safeWrite = function(obj) {
    try {
        const data = typeof obj === 'string' ? obj : JSON.stringify(obj);
        res.write('data: ' + data + '\n\n');
    } catch (e) {
        console.error('Stream write error:', e.message);
    }
};

const processData = function(rawData) {
    if (!rawData || rawData.trim() === '') return;

    if (rawData.trim() === '[DONE]') {
        safeWrite('[DONE]');
        return;
    }

    try {
        const parsed = JSON.parse(rawData);
        const delta = parsed && parsed.choices && parsed.choices[0] ? parsed.choices[0].delta : null;

        if (delta && SHOW_REASONING) {
            const reasoning = delta.reasoning_content;
            const content = delta.content;

            if (reasoning) {
                delta.content = reasoningActive ? reasoning : '\u003Cthink\u003E\n' + reasoning;
                reasoningActive = true;
                delete delta.reasoning_content;
            } else if (content && reasoningActive) {
                delta.content = '\n\u003C/think\u003E\n\n' + content;
                reasoningActive = false;
            }
        }

        safeWrite(parsed);
    } catch (e) {
        partialData += rawData;
        try {
            const parsed = JSON.parse(partialData);
            partialData = '';

            const delta = parsed && parsed.choices && parsed.choices[0] ? parsed.choices[0].delta : null;
            if (delta && SHOW_REASONING) {
                const reasoning = delta.reasoning_content;
                const content = delta.content;

                if (reasoning) {
                    delta.content = reasoningActive ? reasoning : '\u003Cthink\u003E\n' + reasoning;
                    reasoningActive = true;
                    delete delta.reasoning_content;
                } else if (content && reasoningActive) {
                    delta.content = '\n\u003C/think\u003E\n\n' + content;
                    reasoningActive = false;
                }
            }

            safeWrite(parsed);
        } catch (e2) {
            if (partialData.length > 100000) {
                console.error('Partial data buffer exceeded, resetting');
                partialData = '';
            }
        }
    }
};

inputStream.on('data', function(chunk) {
    buffer += chunk.toString('utf8');
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || '';

    for (let i = 0; i < lines.length; i++) {
        if (!lines[i].startsWith('data: ')) continue;
        const dataStr = lines[i].slice(6);
        processData(dataStr);
    }
});

inputStream.on('end', function() {
    if (buffer.startsWith('data: ')) {
        processData(buffer.slice(6));
    }

    if (reasoningActive) {
        safeWrite({
            choices: [{ delta: { content: '\n\u003C/think\u003E' } }]
        });
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
const openaiResponse = {
id: 'chatcmpl-' + Date.now(),
object: 'chat.completion',
created: Math.floor(Date.now() / 1000),
model: model,
choices: (data.choices || []).map(function(choice, index) {
let fullContent = (choice && choice.message && choice.message.content) || '';

            if (SHOW_REASONING && choice && choice.message && choice.message.reasoning_content) {
                fullContent = '\u003Cthink\u003E\n' + choice.message.reasoning_content + '\n\u003C/think\u003E\n\n' + fullContent;
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
console.log(' - SHOW_REASONING: ' + SHOW_REASONING);
console.log(' - ENABLE_THINKING_MODE: ' + ENABLE_THINKING_MODE);
console.log(' - REQUEST_TIMEOUT: ' + (REQUEST_TIMEOUT / 1000) + 's');
console.log(' - Frankenstein preset loaded: ' + (PRESET_FRANKENSTEIN ? 'YES' : 'NO'));
console.log(' - FranKIMstein preset loaded: ' + (PRESET_FRANKIMSTEIN ? 'YES' : 'NO'));

if (!NIM_API_KEY) {
    console.warn('WARNING: NIM_API_KEY is missing!');
}

console.log('');
console.log('Model -> Preset Mapping:');
const modelKeys = Object.keys(MODEL_MAPPING);
for (let i = 0; i < modelKeys.length; i++) {
    const openaiId = modelKeys[i];
    const nimId = MODEL_MAPPING[openaiId];
    const preset = getPresetForModel(nimId);
    const presetName = preset ? preset.name : 'NONE';
    const isKimi = isKimiModel(nimId) ? 'Kimi' : 'Non-Kimi';
    console.log('   - ' + openaiId + ' -> ' + nimId + ' (' + isKimi + ') -> ' + presetName);
}

});
