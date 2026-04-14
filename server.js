const express = require('express');
const cors = require('cors');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
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

// --- Preset Loading ---
const PRESETS_DIR = path.join(__dirname, 'presets');

function loadPreset(presetName) {
    const filePath = path.join(PRESETS_DIR, `${presetName}.json`);
    try {
        const raw = fs.readFileSync(filePath, 'utf8');
        return JSON.parse(raw);
    } catch (err) {
        console.warn(`⚠️ Could not load preset "${presetName}": ${err.message}`);
        return null;
    }
}

const PRESET_FRANKENSTEIN = loadPreset('frankenstein');
const PRESET_FRANKIMSTEIN = loadPreset('frankimstein');

// Model mapping: OpenAI-compatible ID -> NIM model ID
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

// --- Preset Selection Logic ---
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
        .filter(p => p.content && p.content.trim() !== '')
        .map(p => ({
            role: p.role || 'system',
            content: p.content.trim()
        }));

    const existingSystemMsgs = originalMessages.filter(m => m.role === 'system');
    const nonSystemMsgs = originalMessages.filter(m => m.role !== 'system');

    return [
        ...existingSystemMsgs,
        ...presetMessages.filter(m => m.role === 'system'),
        ...presetMessages.filter(m => m.role !== 'system'),
        ...nonSystemMsgs
    ];
}

// --- Middleware ---
app.use(cors());
app.use(express.json({ limit: '5mb' }));
app.use(express.urlencoded({ limit: '5mb', extended: true }));

// --- Presets Endpoint ---
app.get('/v1/presets', (req, res) => {
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
    res.json({ presets });
});

// --- Helpers ---
const toBoolean = (val) => val === true || val === 'true';

const getEnhancedMessages = (model, messages) => {
    const formattingNudge = {
        role: 'system',
        content: 'CRITICAL INSTRUCTION: Use Markdown. ALWAYS use double line breaks (\\n\\n) between paragraphs. No walls of text.'
    };

    const hasFormattingInstruction = messages.some(
        msg => msg.role === 'system' &&
            (msg.content.includes('Markdown') ||
             msg.content.includes('paragraph') ||
             msg.content.includes('formatting') ||
             msg.content.includes('CRITICAL INSTRUCTION'))
    );

    let enhanced;
    if (hasFormattingInstruction) {
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

    return { temperature: sanitizedTemp ?? 0.7, max_tokens: sanitizedMaxTokens ?? 4096 };
};

// --- Routes ---
app.get('/health', (req, res) => {
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

app.get('/v1/models', (req, res) => {
    const models = Object.keys(MODEL_MAPPING).map(id => {
        const nimModel = MODEL_MAPPING[id];
        const preset = getPresetForModel(nimModel);
        return {
            id,
            object: 'model',
            created: Math.floor(Date.now() / 1000),
            owned_by: 'nvidia-nim-proxy',
            nim_model: nimModel,
            preset: preset ? (preset.name.toLowerCase().includes('kim') ? 'frankimstein' : 'frankenstein') : 'none'
        };
    });
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

        const { model, messages, temperature, max_tokens, stream, preset_override } = req.body;

        if (!messages || !Array.isArray(messages) || messages.length === 0) {
            return res.status(400).json({
                error: {
                    message: 'Missing or invalid messages array',
                    code: 400
                }
            });
        }

        const { temperature: sanitizedTemp, max_tokens: sanitizedMaxTokens } =
            validateAndSanitizeParams(temperature, max_tokens);

        const wantsStream = toBoolean(stream);
        const nimModel = MODEL_MAPPING[model] || model;

        // --- PRESET INJECTION ---
        let preset;
        if (preset_override && (preset_override === 'frankenstein' || preset_override === 'frankimstein')) {
            preset = preset_override === 'frankimstein' ? PRESET_FRANKIMSTEIN : PRESET_FRANKENSTEIN;
            console.log(`📋 Preset override: ${preset_override} (forced by client)`);
        } else {
            preset = getPresetForModel(nimModel);
        }

        let processedMessages = messages;

        if (preset) {
            processedMessages = buildOrderedMessagesFromPreset(preset, messages);
            console.log(`📋 Preset applied: ${preset.name} for model ${nimModel}`);
            console.log(`   - Preset prompts injected: ${preset.prompts.length}`);
        } else {
            console.log(`⚠️ No preset available for model ${nimModel}, using raw messages`);
        }

        const enhancedMessages = getEnhancedMessages(nimModel, processedMessages);
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
    res.setHeader('X-Accel-Buffering', 'no');

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
                    delta.content = reasoningActive ? reasoning : `\n\n${content}`;
                    reasoningActive = false;
                }
            }

            safeWrite(parsed);
        } catch (e) {
            partialData += rawData;
            try {
                const parsed = JSON.parse(partialData);
                partialData = '';

                const delta = parsed?.choices?.[0]?.delta;
                if (delta && SHOW_REASONING) {
                    const reasoning = delta.reasoning_content;
                    const content = delta.content;

                    if (reasoning) {
                        delta.content = reasoningActive ? reasoning : `\n\n${content}`;
                        reasoningActive = false;
                    }
                }

                safeWrite(parsed);
            } catch {
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
        if (buffer.startsWith('data: ')) {
            processData(buffer.slice(6));
        }

        if (reasoningActive) {
            safeWrite({
                choices: [{
                    delta: { content: '\n' }
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
                    fullContent = `\n\n${fullContent}`;
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
    console.log(`   - Frankenstein preset loaded: ${PRESET_FRANKENSTEIN ? '✅' : '❌'}`);
    console.log(`   - FranKIMstein preset loaded: ${PRESET_FRANKIMSTEIN ? '✅' : '❌'}`);

    if (!NIM_API_KEY) {
        console.warn('⚠️ WARNING: NIM_API_KEY is missing!');
    }

    // Log model-to-preset mapping
    console.log('\n📋 Model → Preset Mapping:');
    for (const [openaiId, nimId] of Object.entries(MODEL_MAPPING)) {
        const preset = getPresetForModel(nimId);
        const presetName = preset ? preset.name : 'NONE';
        const isKimi = isKimiModel(nimId) ? '🌙 Kimi' : '🧟 Non-Kimi';
        console.log(`   - ${openaiId} → ${nimId} (${isKimi}) → ${presetName}`);
    }
});
```

---

## GitHub Deployment Tutorial

<details>
<summary>📁 Step-by-step: Push to GitHub & Deploy</summary>

### 1. Local Project Setup

```bash
# Navigate to your project
cd /path/to/your/project

# Ensure this folder structure exists:
# ├── server.js
# ├── package.json
# ├── presets/
# │   ├── frankenstein.json
# │   └── frankimstein.json

# If presets folder doesn't exist:
mkdir -p presets
```

### 2. Save the Preset Files

Save the two JSON files from earlier into the `presets/` folder:
- `presets/frankenstein.json` — the full CoT preset
- `presets/frankimstein.json` — the chill Kimi preset

### 3. Update `package.json` (if needed)

Make sure `package.json` has these dependencies:

```json
{
  "name": "nvidia-nim-proxy",
  "version": "1.0.0",
  "main": "server.js",
  "scripts": {
    "start": "node server.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "axios": "^1.6.0"
  }
}
```

### 4. Create `.gitignore`

```gitignore
node_modules/
.env
*.log
```

### 5. Create `.env` (local only, never commit this)

```env
NIM_API_KEY=nvapi-xxxxxxxxxxxxx
SHOW_REASONING=true
ENABLE_THINKING_MODE=true
REQUEST_TIMEOUT=600000
PORT=3000
```

### 6. Git Init & Push

```bash
# Initialize git (if not already)
git init

# Add all files
git add server.js package.json presets/ .gitignore

# Commit
git commit -m "Add Freaky Frankenstein / FranKIMstein preset integration"

# Add remote (replace with your actual repo URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push
git push -u origin main
```

### 7. Deploy (Choose One)

<details>
<summary>🔵 Deploy to Render</summary>

1. Go to [render.com](https://render.com) → **New** → **Web Service**
2. Connect your GitHub repo
3. Settings:
   - **Build Command:** `npm install`
   - **Start Command:** `node server.js`
4. Add **Environment Variables:**
   - `NIM_API_KEY` = your key
   - `SHOW_REASONING` = `true`
   - `ENABLE_THINKING_MODE` = `true`
5. Deploy!

</details>

<details>
<summary>▲ Deploy to Vercel (serverless)</summary>

Create `vercel.json`:

```json
{
  "version": 2,
  "builds": [
    { "src": "server.js", "use": "@vercel/node" }
  ],
  "routes": [
    { "src": "/(.*)", "dest": "server.js" }
  ]
}
```

```bash
npm i -g vercel
vercel
```

Add env vars in Vercel dashboard.

</details>

<details>
<summary>🚂 Deploy to Railway</summary>

1. Go to [railway.app](https://railway.app)
2. **New Project** → **Deploy from GitHub**
3. Select your repo
4. Add **Variables:**
   - `NIM_API_KEY`
   - `SHOW_REASONING`
   - `ENABLE_THINKING_MODE`
5. It auto-detects Node.js and runs `npm start`

</details>

### 8. Verify

After deployment, hit these endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Shows server status + preset load state |
| `GET /v1/models` | Lists models with their assigned preset |
| `GET /v1/presets` | Lists available presets |
| `POST /v1/chat/completions` | Chat with preset auto-injection |

Test with `curl`:

```bash
# Health check
curl https://your-app.onrender.com/health

# Check which preset each model gets
curl https://your-app.onrender.com/v1/models

# Test Kimi model (should get FranKIMstein)
curl -X POST https://your-app.onrender.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'

# Test non-Kimi model (should get Frankenstein)
curl -X POST https://your-app.onrender.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'

# Force a specific preset with override
curl -X POST https://your-app.onrender.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "preset_override": "frankimstein",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

</details>

---

## Summary of Changes

| Component | What Changed |
|---|---|
| **`presets/frankenstein.json`** | NEW — Full CoT preset for non-Kimi models |
| **`presets/frankimstein.json`** | NEW — Chill immediate-response preset for Kimi models |
| **`server.js`** | Added `fs`/`path` imports, preset loading, `isKimiModel()`, `getPresetForModel()`, `buildOrderedMessagesFromPreset()`, `/v1/presets` endpoint, `preset_override` support in chat endpoint, enhanced logging |
| **Model→Preset Logic** | `moonshotai/*` or `*kimi*` → FranKIMstein, everything else → Frankenstein |

### Key Design Decisions

- **Preset JSONs are loaded at startup** — no disk I/O per request
- **Preset injection preserves client system messages** — your front-end's system prompt isn't overwritten, it's prepended
- **`preset_override` field** — lets you force a specific preset regardless of model (useful for testing)
- **Kimi detection** — based on `moonshotai` or `kimi` in the NIM model ID string
- **FranKIMstein omits Chain of Thought** — the entire point is Kimi models respond immediately without the 8-step Chinese CoT
