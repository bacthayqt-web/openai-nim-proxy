const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { StringDecoder } = require('string_decoder');

const app = express();
const PORT = process.env.PORT || 3000;

// --- Configuration ---
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const SHOW_REASONING = true;
const ENABLE_THINKING_MODE = true;

const MODEL_MAPPING = {
    'gpt-3.5-turbo': 'moonshotai/kimi-k2.5',
    'gpt-4': 'z-ai/glm5',
    'gpt-4-turbo': 'deepseek-ai/deepseek-v3.1',
    'gpt-4o': 'deepseek-ai/deepseek-v3.2',
    'gpt-4-0613': 'mistralai/mistral-large-3-675b-instruct-2512',
    'claude-3-opus': 'moonshotai/kimi-k2-thinking',
    'claude-3-sonnet': 'z-ai/glm4.7',
    'gemini-pro': 'deepseek-ai/deepseek-v3.1-terminus'
};

// --- Middleware ---
app.use(cors());
app.use(express.json({ limit: '5mb' }));
app.use(express.urlencoded({ limit: '5mb', extended: true }));

// --- Helpers ---
const getEnhancedMessages = (model, messages) => {
    const formattingNudge = {
        role: 'system',
        content: 'CRITICAL INSTRUCTION: Use Markdown. ALWAYS use double line breaks (\\n\\n) between paragraphs. No walls of text.'
    };

    let enhanced = [formattingNudge, ...messages];

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

// --- Routes ---
app.get('/health', (req, res) => res.json({ status: 'ok', reasoning_display: SHOW_REASONING }));

app.get('/v1/models', (req, res) => {
    const models = Object.keys(MODEL_MAPPING).map(id => ({ id, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy' }));
    res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
    try {
        const { model, messages, temperature, max_tokens, stream } = req.body;
        if (!messages) return res.status(400).json({ error: "Missing messages array" });

        const nimModel = MODEL_MAPPING[model] || model;
        const enhancedMessages = getEnhancedMessages(nimModel, messages);
        const supportsThinking = nimModel.includes('deepseek') || nimModel.includes('thinking');

        const nimRequest = {
            model: nimModel,
            messages: enhancedMessages,
            temperature: temperature ?? 0.7,
            max_tokens: max_tokens ?? 4096,
            extra_body: (ENABLE_THINKING_MODE && supportsThinking) ? { chat_template_kwargs: { thinking: true } } : undefined,
            stream: stream || false,
            ...(nimModel.includes('deepseek') && {
                top_p: 0.9,
                frequency_penalty: 0.1,
                presence_penalty: 0.1
            })
        };

        console.log(`Processing request for model: ${nimModel}`);
        console.log('Request body:', JSON.stringify(nimRequest, null, 2));

        const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
            headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
            responseType: stream ? 'stream' : 'json',
            timeout: 600000
        });

        if (stream) {
            handleStream(response.data, res, nimModel);
        } else {
            handleNonStream(response.data, model, res);
        }
    } catch (error) {
        console.error('Proxy error:', error.response?.data || error.message);
        if (!res.headersSent) {
            res.status(error.response?.status || 500).json({
                error: { message: error.message, code: error.response?.status || 500 }
            });
        }
    }
});

function handleStream(inputStream, res, nimModel) {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    let buffer = '';
    let reasoningActive = false;
    const decoder = new StringDecoder('utf8');
    let firstChunk = true;
    const isThinkingModel = nimModel.includes('thinking') || nimModel.includes('deepseek');

    inputStream.on('data', (chunk) => {
        buffer += decoder.write(chunk);
        let lines = buffer.split(/\r?\n/);
        buffer = lines.pop() || '';

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const dataStr = line.slice(6).trim();
                if (dataStr === '[DONE]') {
                    res.write('data: [DONE]\n\n');
                    continue;
                }

                try {
                    const data = JSON.parse(dataStr);
                    const delta = data.choices?.[0]?.delta;

                    if (delta && SHOW_REASONING) {
                        const reasoning = delta.reasoning_content;
                        const content = delta.content;

                        if (reasoning) {
                            if (!reasoningActive) {
                                // For thinking models, don't include reasoning in main content
                                if (isThinkingModel) {
                                    delta.content = '';
                                } else {
                                    delta.content = `<think>\n${reasoning}`;
                                }
                                reasoningActive = true;
                            } else {
                                // For thinking models, don't include reasoning in main content
                                if (isThinkingModel) {
                                    delta.content = '';
                                } else {
                                    delta.content = reasoning;
                                }
                            }
                            delete delta.reasoning_content;
                        } else if (content && reasoningActive) {
                            // For thinking models, don't add think tags to main content
                            if (isThinkingModel) {
                                delta.content = content;
                            } else {
                                delta.content = `\n</think>\n\n${content}`;
                            }
                            reasoningActive = false;
                        }
                    }

                    // For GLM models, ensure we don't lose the first few characters
                    if (firstChunk && nimModel.includes('glm')) {
                        setTimeout(() => {
                            res.write(`data: ${JSON.stringify(data)}\n\n`);
                            firstChunk = false;
                        }, 10);
                    } else {
                        res.write(`data: ${JSON.stringify(data)}\n\n`);
                    }
                } catch (e) {
                    console.error('Error parsing stream data:', e);
                }
            }
        }
    });

    inputStream.on('end', () => {
        if (reasoningActive) {
            res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: '\n</think>' } }] })}\n\n`);
        }
        res.end();
    });
}

function handleNonStream(data, model, res) {
    console.log('Raw response data:', JSON.stringify(data, null, 2));

    // Handle empty responses
    if (!data.choices || data.choices.length === 0) {
        return res.status(500).json({
            error: {
                message: "Empty response from model",
                code: 500
            }
        });
    }

    const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: data.choices.map(choice => {
            let fullContent = choice.message?.content || '';
            let reasoningContent = '';

            if (SHOW_REASONING && choice.message?.reasoning_content) {
                reasoningContent = choice.message.reasoning_content;
                // Don't include reasoning in the main content for Chub AI
                if (model.includes('thinking') || model.includes('deepseek')) {
                    fullContent = fullContent; // Keep original content
                } else {
                    fullContent = `<think>\n${reasoningContent}\n</think>\n\n${fullContent}`;
                }
            }

            return {
                index: choice.index,
                message: {
                    role: choice.message.role,
                    content: fullContent,
                    ...(reasoningContent && { reasoning_content: reasoningContent })
                },
                finish_reason: choice.finish_reason
            };
        }),
        usage: data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    };

    console.log('Final response:', JSON.stringify(openaiResponse, null, 2));
    res.json(openaiResponse);
}

app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 Proxy running on port ${PORT}`);
    if (!NIM_API_KEY) console.warn('⚠️ WARNING: NIM_API_KEY is missing!');
});
