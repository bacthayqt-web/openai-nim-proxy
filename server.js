const express = require('express');
const cors = require('cors');
const axios = require('axios');
const app = express();
const PORT = process.env.PORT || 3000;

// --- Configuration ---
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const SHOW_REASONING = process.env.SHOW_REASONING !== 'false';
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE !== 'false';
const REQUEST_TIMEOUT = parseInt(process.env.REQUEST_TIMEOUT || '600000', 10);
const MAX_TEMPERATURE = 2.0;
const MAX_MAX_TOKENS = 128000;

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

// --- Middleware ---
app.use(cors());
app.use(express.json({ limit: '5mb' }));
app.use(express.urlencoded({ limit: '5mb', extended: true }));

// --- Helpers ---
const toBoolean = (val) => val === true || val === 'true';

const getEnhancedMessages = (model, messages) => {
    const formattingNudge = {
        role: 'system',
        content: 'CRITICAL INSTRUCTION: Use Markdown. ALWAYS use double line breaks (\\n\\n) between paragraphs. No walls of text.'
    };

    // Check if a similar formatting system message already exists to avoid duplicates
    const hasFormattingInstruction = messages.some(
        msg => msg.role === 'system' &&
            (msg.content.includes('Markdown') ||
             msg.content.includes('paragraph') ||
             msg.content.includes('formatting') ||
             msg.content.includes('CRITICAL INSTRUCTION'))
    );

    let enhanced;
    if (hasFormattingInstruction) {
        // Merge existing system messages to avoid duplication
        enhanced = messages.map(msg => {
            if (msg.role === 'system' &&
                (msg.content.includes('Markdown') ||
                 msg.content.includes('paragraph') ||
                 msg.content.includes('formatting'))) {
                return {
                    ...msg,
                    content: `${formattingNudge.content}\n\n${msg.content}`
                };
            }
            return msg;
        });
    } else {
        enhanced = [formattingNudge, ...messages];
    }

    // Add formatting hint for GLM models
    if (model.includes('glm')) {
        const lastIndex = enhanced.length - 1;
        if (lastIndex >= 0 && enhanced[lastIndex].role === 'user') {
            enhanced[lastIndex] = {
                ...enhanced[lastIndex],
                content: `${enhanced[lastIndex].content}\n\n[Formatting Rule: Use clear, separate paragraphs with double line breaks.]`
            };
        }
    }

    return enhanced;
};

const validateAndSanitizeParams = (temperature, max_tokens) => {
    let sanitizedTemp = temperature;
    if (temperature !== undefined && temperature !== null) {
        sanitizedTemp = Math.max(0, Math.min(MAX_TEMPERATURE, parseFloat(temperature)));
        if (isNaN(sanitizedTemp)) {
            sanitizedTemp = 0.7; // default
        }
    }

    let sanitizedMaxTokens = max_tokens;
    if (max_tokens !== undefined && max_tokens !== null) {
        sanitizedMaxTokens = Math.min(MAX_MAX_TOKENS, Math.max(1, parseInt(max_tokens, 10)));
        if (isNaN(sanitizedMaxTokens)) {
            sanitizedMaxTokens = 4096; // default
        }
    }

    return { temperature: sanitizedTemp ?? 0.7, max_tokens: sanitizedMaxTokens ?? 4096 };
};

// --- Routes ---
app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        reasoning_display: SHOW_REASONING,
        thinking_mode: ENABLE_THINKING_MODE,
        timeout_seconds: REQUEST_TIMEOUT / 1000
    });
});

app.get('/v1/models', (req, res) => {
    const models = Object.keys(MODEL_MAPPING).map(id => ({
        id,
        object: 'model',
        created: Math.floor(Date.now() / 1000),
        owned_by: 'nvidia-nim-proxy'
    }));
    res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
    try {
        if (!NIM_API_KEY) {
            return res.status(500).json({
                error: {
                    message: 'NIM_API_KEY missing',
                    code: 500
                }
            });
        }

        const { model, messages, temperature, max_tokens, stream } = req.body;

        // Validate messages
        if (!messages || !Array.isArray(messages) || messages.length === 0) {
            return res.status(400).json({
                error: {
                    message: 'Missing or invalid messages array',
                    code: 400
                }
            });
        }

        // Validate and sanitize numeric parameters
        const { temperature: sanitizedTemp, max_tokens: sanitizedMaxTokens } =
            validateAndSanitizeParams(temperature, max_tokens);

        const wantsStream = toBoolean(stream);
        const nimModel = MODEL_MAPPING[model] || model;
        const enhancedMessages = getEnhancedMessages(nimModel, messages);
        const supportsThinking = nimModel.includes('deepseek') || nimModel.includes('thinking');

        const nimRequest = {
            model: nimModel,
            messages: enhancedMessages,
            temperature: sanitizedTemp,
            max_tokens: sanitizedMaxTokens,
            stream: wantsStream,
            ...(ENABLE_THINKING_MODE && supportsThinking ? {
                extra_body: {
                    chat_template_kwargs: { thinking: true }
                }
            } : {})
        };

        const response = await axios.post(
            `${NIM_API_BASE}/chat/completions`,
            nimRequest,
            {
                headers: {
                    Authorization: `Bearer ${NIM_API_KEY}`,
                    'Content-Type': 'application/json'
                },
                responseType: wantsStream ? 'stream' : 'json',
                timeout: REQUEST_TIMEOUT,
                validateStatus: () => true
            }
        );

        if (response.status >= 400) {
            // Avoid double-sending if already headers sent
            if (res.headersSent) return;

            const errorMessage = response.data?.error?.message ||
                response.data?.error?.code ||
                'Upstream error';

            return res.status(response.status).json({
                error: {
                    message: errorMessage,
                    code: response.status
                }
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
            status: error.response?.status,
            data: error.response?.data
        });

        if (!res.headersSent) {
            res.status(500).json({
                error: {
                    message: error.message || 'Internal server error',
                    code: 500
                }
            });
        }
    }
});

function handleStream(inputStream, res) {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no'); // Disable nginx buffering if present

    let buffer = '';
    let partialData = '';
    let reasoningActive = false;

    const safeWrite = (obj) => {
        try {
            const data = typeof obj === 'string' ? obj : JSON.stringify(obj);
            res.write(`data: ${data}\n\n`);
        } catch (e) {
            console.error('Stream write error:', e.message);
        }
    };

    const processData = (rawData) => {
        if (!rawData || rawData.trim() === '') return;

        if (rawData.trim() === '[DONE]') {
            safeWrite('[DONE]');
            return;
        }

        try {
            const parsed = JSON.parse(rawData);
            const delta = parsed?.choices?.[0]?.delta;

            if (delta && SHOW_REASONING) {
                const reasoning = delta.reasoning_content;
                const content = delta.content;

                if (reasoning) {
                    delta.content = reasoningActive ? reasoning : `<think>\n${reasoning}`;
                    reasoningActive = true;
                    delete delta.reasoning_content;
                } else if (content && reasoningActive) {
                    delta.content = `\n</think>\n\n${content}`;
                    reasoningActive = false;
                }
            }

            safeWrite(parsed);
        } catch (e) {
            // Accumulate partial data and try to parse when we have more
            partialData += rawData;
            try {
                const parsed = JSON.parse(partialData);
                partialData = '';

                const delta = parsed?.choices?.[0]?.delta;
                if (delta && SHOW_REASONING) {
                    const reasoning = delta.reasoning_content;
                    const content = delta.content;

                    if (reasoning) {
                        delta.content = reasoningActive ? reasoning : `<think>\n${reasoning}`;
                        reasoningActive = true;
                        delete delta.reasoning_content;
                    } else if (content && reasoningActive) {
                        delta.content = `\n</think>\n\n${content}`;
                        reasoningActive = false;
                    }
                }

                safeWrite(parsed);
            } catch {
                // Still incomplete, wait for more data
                // If partialData gets too large, something is wrong - reset
                if (partialData.length > 100000) {
                    console.error('Partial data buffer exceeded, resetting');
                    partialData = '';
                }
            }
        }
    };

    inputStream.on('data', (chunk) => {
        buffer += chunk.toString('utf8');
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop() || '';

        for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const dataStr = line.slice(6);
            processData(dataStr);
        }
    });

    inputStream.on('end', () => {
        // Process any remaining data in buffer
        if (buffer.startsWith('data: ')) {
            processData(buffer.slice(6));
        }

        // Close reasoning block if still open
        if (reasoningActive) {
            safeWrite({
                choices: [{
                    delta: { content: '\n</think>' }
                }]
            });
        }

        safeWrite('[DONE]');
        res.end();
    });

    inputStream.on('error', (err) => {
        console.error('Stream error:', err.message);
        if (!res.headersSent) {
            res.status(500).json({
                error: {
                    message: 'Stream processing error',
                    code: 500
                }
            });
        }
        res.end();
    });
}

function handleNonStream(data, model, res) {
    try {
        const openaiResponse = {
            id: `chatcmpl-${Date.now()}`,
            object: 'chat.completion',
            created: Math.floor(Date.now() / 1000),
            model,
            choices: (data.choices || []).map((choice, index) => {
                let fullContent = choice?.message?.content || '';

                if (SHOW_REASONING && choice?.message?.reasoning_content) {
                    fullContent = `<think>\n${choice.message.reasoning_content}\n</think>\n\n${fullContent}`;
                }

                return {
                    index: choice.index ?? index,
                    message: {
                        role: choice?.message?.role || 'assistant',
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
            error: {
                message: 'Response formatting error',
                code: 500
            }
        });
    }
}

// --- Start Server ---
app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 Proxy running on port ${PORT}`);
    console.log(`   - SHOW_REASONING: ${SHOW_REASONING}`);
    console.log(`   - ENABLE_THINKING_MODE: ${ENABLE_THINKING_MODE}`);
    console.log(`   - REQUEST_TIMEOUT: ${REQUEST_TIMEOUT / 1000}s`);

    if (!NIM_API_KEY) {
        console.warn('⚠️ WARNING: NIM_API_KEY is missing!');
    }
});
