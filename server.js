require('dotenv').config(); // Load .env if present
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { StringDecoder } = require('string_decoder');

const app = express();
const PORT = process.env.PORT || 3000;

// --- Configuration ---
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';

if (!NIM_API_KEY) {
  throw new Error('MISSING NIM_API_KEY environment variable');
}

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'moonshotai/kimi-k2.5',
  'gpt-4': 'z-ai/glm5',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v3.1',
  'gpt-4o': 'deepseek-ai/deepseek-v3.2',
  'claude-3-opus': 'moonshotai/kimi-k2-thinking',
  'claude-3-sonnet': 'z-ai/glm4.7',
  'gemini-pro': 'deepseek-ai/deepseek-v3.1-terminus'
};

// --- Middleware ---
app.use(cors({ origin: process.env.ALLOWED_ORIGINS?.split(',') || '*' }));
app.use(express.json({ limit: process.env.MAX_PAYLOAD || '5mb' }));
app.use(express.urlencoded({ extended: true, limit: process.env.MAX_PAYLOAD || '5mb' }));

// --- Helpers ---
const validateMessages = (messages) => {
  if (!Array.isArray(messages)) throw new Error('Messages must be an array');
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (!msg.role || !['system', 'user', 'assistant', 'tool'].includes(msg.role)) {
      throw new Error(`Invalid role at messages[${i}]`);
    }
    if (typeof msg.content !== 'string' && !Array.isArray(msg.content)) {
      throw new Error(`Invalid content at messages[${i}]`);
    }
  }
};

const getEnhancedMessages = (model, messages) => {
  const formattingNudge = { 
    role: 'system', 
    content: 'CRITICAL INSTRUCTION: Use Markdown. ALWAYS use double line breaks (\\n\\n) between paragraphs. No walls of text.' 
  };
  
  let enhanced = [formattingNudge, ...messages];

  if (model.includes('glm')) {
    const lastIndex = enhanced.length - 1;
    if (enhanced[lastIndex]?.role === 'user') {
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
  const models = Object.keys(MODEL_MAPPING).map(id => ({ 
    id, object: 'model', created: Math.floor(Date.now() / 1000), owned_by: 'nvidia-nim-proxy' 
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, max_completion_tokens, stream } = req.body;
    
    if (!model) return res.status(400).json({ error: 'Missing model' });
    validateMessages(messages);

    const nimModel = MODEL_MAPPING[model] || model;
    const enhancedMessages = getEnhancedMessages(nimModel, messages);
    const supportsThinking = nimModel.includes('deepseek') || nimModel.includes('thinking');

    const nimRequest = {
      model: nimModel,
      messages: enhancedMessages,
      temperature: temperature ?? 0.7,
      max_tokens: max_completion_tokens ?? max_tokens ?? 4096,
      extra_body: (ENABLE_THINKING_MODE && supportsThinking) ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream || false,
      stream_options: stream ? { include_usage: true } : undefined
    };

    const abortController = new AbortController();
    res.on('close', () => abortController.abort());

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: stream ? 'stream' : 'json',
      timeout: 600_000,
      signal: abortController.signal
    });

    if (stream) {
      handleStream(response.data, res);
    } else {
      handleNonStream(response.data, model, res);
    }
  } catch (error) {
    if (axios.isCancel(error)) return; // Client disconnected
    console.error('Proxy error:', error.response?.data || error.message);
    if (!res.headersSent) {
      res.status(error.response?.status || 502).json({ 
        error: { message: 'Upstream request failed', code: error.response?.status || 502 } 
      });
    }
  }
});

function handleStream(inputStream, res) {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no'); // Disable proxy buffering

  let buffer = '';
  let reasoningActive = false;
  const decoder = new StringDecoder('utf8');

  const safeWrite = (data) => {
    if (!res.writableEnded) res.write(data);
  };

  inputStream.on('data', (chunk) => {
    buffer += decoder.write(chunk);
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const dataStr = line.slice(6).trim();
      if (dataStr === '[DONE]') {
        safeWrite('data: [DONE]\n\n');
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
        safeWrite(`data: ${JSON.stringify(data)}\n\n`);
      } catch (e) {
        // Log malformed lines in debug mode, ignore otherwise
        if (process.env.NODE_ENV === 'development') console.warn('Malformed SSE chunk:', line);
      }
    }
  });

  inputStream.on('end', () => {
    if (reasoningActive && !res.writableEnded) {
      safeWrite(`data: ${JSON.stringify({ choices: [{ delta: { content: '\n</think>' } }] })}\n\n`);
    }
    if (!res.writableEnded) safeWrite('data: [DONE]\n\n');
    res.end();
  });

  inputStream.on('error', (err) => {
    console.error('Stream error:', err.message);
    if (!res.writableEnded) {
      safeWrite('data: [DONE]\n\n');
      res.end();
    }
  });
}

function handleNonStream(data, model, res) {
  const openaiResponse = {
    id: `chatcmpl-${Date.now()}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: data.choices.map(choice => {
      let fullContent = choice.message?.content || '';
      if (SHOW_REASONING && choice.message?.reasoning_content) {
        fullContent = `<think>\n${choice.message.reasoning_content}\n</think>\n\n${fullContent}`;
      }
      return {
        index: choice.index ?? 0,
        message: { role: choice.message?.role || 'assistant', content: fullContent },
        finish_reason: mapFinishReason(choice.finish_reason)
      };
    }),
    usage: data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
  };
  res.json(openaiResponse);
}

function mapFinishReason(reason) {
  if (!reason) return 'stop';
  return { eos: 'stop', stop: 'stop', length: 'length', tool_calls: 'tool_calls' }[reason] || 'stop';
}

app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 NIM Proxy running on port ${PORT}`);
  console.log(`🔑 API Key: ${NIM_API_KEY ? '✅ Loaded' : '❌ Missing'}`);
  console.log(`🧠 Thinking Mode: ${ENABLE_THINKING_MODE ? 'Enabled' : 'Disabled'}`);
});
