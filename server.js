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
    'gpt-4-0613': 'moonshotai/kimi-k2-instruct',
    'claude-3-opus': 'moonshotai/kimi-k2-thinking',
    'claude-3-sonnet': 'z-ai/glm4.7',
    'gemini-pro': 'deepseek-ai/deepseek-v3.1-terminus'
};

// --- Middleware ---
app.use(cors()); 
app.use(express.json({ limit: '5mb' })); // Reduced for security
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
            stream: stream || false
        };

        const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
            headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
            responseType: stream ? 'stream' : 'json',
            timeout: 600000
        });

        if (stream) {
            handleStream(response.data, res);
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

function handleStream(inputStream, res) {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    let buffer = '';
    let reasoningActive = false;
    const decoder = new StringDecoder('utf8');

    inputStream.on('data', (chunk) => {
        buffer += decoder.write(chunk);
        let lines = buffer.split(/\r?\n/); // Fix: Handle CRLF
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
                                delta.content = `<think>\n${reasoning}`;
                                reasoningActive = true;
                            } else {
                                delta.content = reasoning;
                            }
                            delete delta.reasoning_content;
                        } else if (content && reasoningActive) {
                            delta.content = `\n</think>\n\n${content}`;
                            reasoningActive = false;
                        }
                    }
                    res.write(`data: ${JSON.stringify(data)}\n\n`);
                } catch (e) {
                    // Silently skip malformed JSON lines
                }
            }
        }
    });

    inputStream.on('end', () => {
        if (reasoningActive) {
            // Close think tag if the stream ended while still thinking
            res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: '\n</think>' } }] })}\n\n`);
        }
        res.end();
    });
}

function handleNonStream(data, model, res) {
    const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: data.choices.map(choice => {
            let fullContent = choice.message?.content || '';
            if (SHOW_REASONING && choice.message?.reasoning_content) {
                fullContent = `<think>\n${choice.message.reasoning_content}\n</think>\n\n${fullContent}`;
            }
            return {
                index: choice.index,
                message: { role: choice.message.role, content: fullContent },
                finish_reason: choice.finish_reason
            };
        }),
        usage: data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    };
    res.json(openaiResponse);
}

app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 Proxy running on port ${PORT}`);
    if (!NIM_API_KEY) console.warn('⚠️ WARNING: NIM_API_KEY is missing!');
});
