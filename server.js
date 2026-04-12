const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// --- Configuration ---
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Safer defaults
const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = true;

// Model mapping
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
function enhanceMessages(messages) {
  return [
    {
      role: 'system',
      content:
        'Format responses using Markdown. Use clear paragraphs with double line breaks.'
    },
    ...messages
  ];
}

function supportsThinking(model) {
  return model.includes('deepseek') || model.includes('thinking');
}

// --- Routes ---
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    reasoning_enabled: SHOW_REASONING
  });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object: 'model',
    created: Date.now(),
    owned_by: 'nim-proxy'
  }));

  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    if (!NIM_API_KEY) {
      return res.status(500).json({
        error: { message: 'NIM_API_KEY is missing', code: 500 }
      });
    }

    const {
      model,
      messages,
      temperature,
      max_tokens
    } = req.body;

    const wantsStream = req.body.stream === true;

    if (!messages) {
      return res.status(400).json({
        error: { message: 'Missing messages array', code: 400 }
      });
    }

    const nimModel = MODEL_MAPPING[model] || model;

    const requestBody = {
      model: nimModel,
      messages: enhanceMessages(messages),
      temperature: temperature ?? 0.7,
      max_tokens: max_tokens ?? 4096,
      stream: wantsStream,
      extra_body:
        ENABLE_THINKING_MODE && supportsThinking(nimModel)
          ? { chat_template_kwargs: { thinking: true } }
          : undefined
    };

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      requestBody,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: wantsStream ? 'stream' : 'json',
        timeout: 600000
      }
    );

    if (wantsStream) {
      return pipeStream(response, res);
    }

    return handleJSON(response.data, model, res);

  } catch (error) {
    console.error('Proxy error:', error.response?.data || error.message);

    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: {
          message: error.response?.data?.error || error.message,
          code: error.response?.status || 500
        }
      });
    }
  }
});

// --- Streaming (SAFE PASSTHROUGH) ---
function pipeStream(upstream, res) {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  upstream.data.on('data', chunk => {
    res.write(chunk); // no mutation = stable
  });

  upstream.data.on('end', () => {
    res.end();
  });

  upstream.data.on('error', err => {
    console.error('Stream error:', err.message);
    res.end();
  });
}

// --- Non-stream response ---
function handleJSON(data, model, res) {
  const response = {
    id: `chatcmpl-${Date.now()}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: data.choices.map(choice => {
      let content = choice.message?.content || '';

      // Optional reasoning injection (safe + explicit)
      if (SHOW_REASONING && choice.message?.reasoning_content) {
        content =
          `<think>\n${choice.message.reasoning_content}\n</think>\n\n` +
          content;
      }

      return {
        index: choice.index,
        message: {
          role: choice.message.role,
          content
        },
        finish_reason: choice.finish_reason
      };
    }),
    usage: data.usage || {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0
    }
  };

  res.json(response);
}

// --- Start server ---
app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Proxy running on port ${PORT}`);

  if (!NIM_API_KEY) {
    console.warn('⚠️ NIM_API_KEY is not set');
  }
});
