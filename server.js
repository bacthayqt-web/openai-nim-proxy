// server.js - OpenAI to NVIDIA NIM API Proxy (Enhanced Formatting Version)
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { StringDecoder } = require('string_decoder'); 

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = true; 
const ENABLE_THINKING_MODE = true; 

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'moonshotai/kimi-k2.5',
  'gpt-4': 'z-ai/glm5',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v3.1',
  'gpt-4o': 'deepseek-ai/deepseek-v3.2',
  'claude-3-opus': 'moonshotai/kimi-k2-thinking',
  'claude-3-sonnet': 'z-ai/glm4.7',
  'gemini-pro': 'deepseek-ai/deepseek-v3.1-terminus' 
};

app.get('/health', (req, res) => {
  res.json({ status: 'ok', reasoning_display: SHOW_REASONING });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // 1. Smart model selection
    let nimModel = MODEL_MAPPING[model] || model; 
    
    // 2. ENHANCEMENT: Force formatting and paragraph breaks
    // Prepending a system message ensures the model prioritizes readability.
    const formattingNudge = { 
        role: 'system', 
        content: 'Format your response using Markdown. Use double line breaks between paragraphs for readability.' 
    };
    const enhancedMessages = [formattingNudge, ...messages];

    // 3. Update the filter to include GLM
const isDeepseek = nimModel.includes('deepseek');
const isGLM = nimModel.includes('glm');
const supportsThinking = isDeepseek || isGLM || nimModel.includes('thinking');

// 4. Use a more robust extra_body
const nimRequest = {
  model: nimModel,
  messages: enhancedMessages,
  temperature: temperature || 0.7,
  max_tokens: max_tokens || 4096, 
  extra_body: (ENABLE_THINKING_MODE && supportsThinking) 
    ? { 
        chat_template_kwargs: { thinking: true }, // Compatibility for older NIMs
        reasoning: true,                         // Required for DeepSeek v3.2+
        enable_reasoning: true                   // Fallback for GLM 5 
      } 
    : undefined,
  stream: stream || false
};
    
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 600000 
    });
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      const decoder = new StringDecoder('utf8'); 
      
      response.data.on('data', (chunk) => {
        buffer += decoder.write(chunk);
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    // Added extra newlines here to ensure separation between reasoning and answer
                    combinedContent += '\n</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => {
        if (!res.writableEnded) res.end();
      });
      
      req.on('close', () => {
        if (!response.data.destroyed) response.data.destroy();
      });

    } else {
      // Non-streaming response logic
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
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
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message);
    if (res.headersSent) {
        res.end();
        return;
    }
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        code: error.response?.status || 500
      }
    });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  if (!NIM_API_KEY) console.warn('⚠️ WARNING: NIM_API_KEY is missing!');
});
