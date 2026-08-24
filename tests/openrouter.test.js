'use strict';

process.env.SHOW_REASONING = 'true';
process.env.ENABLE_THINKING_MODE = 'true';
process.env.REASONING_EFFORT = 'high';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const helpers = require('../server')._test;

assert.strictEqual(
    helpers.detectProvider({ path: '/openrouter/v1/chat/completions' }),
    'openrouter'
);
assert.strictEqual(
    helpers.detectProvider({ path: '/janitor/openrouter/v1/chat/completions' }),
    'openrouter'
);
assert.strictEqual(
    helpers.detectProvider({ path: '/v1/chat/completions' }),
    'nim'
);
assert.strictEqual(
    helpers.detectFrontend({ path: '/janitor/openrouter/v1/chat/completions' }),
    'janitor'
);

assert.strictEqual(
    helpers.resolveOpenRouterModel('anthropic/example-model', {
        defaultModel: 'openai/fallback-model',
        mapping: {}
    }),
    'anthropic/example-model',
    'Canonical OpenRouter IDs must pass through unchanged'
);
assert.strictEqual(
    helpers.resolveOpenRouterModel('gpt-4', {
        defaultModel: 'openai/fallback-model',
        mapping: { 'gpt-4': 'google/mapped-model' }
    }),
    'google/mapped-model',
    'Explicit aliases must beat the default model'
);
assert.strictEqual(
    helpers.resolveOpenRouterModel('gpt-4', {
        defaultModel: 'openai/fallback-model',
        mapping: {}
    }),
    'openai/fallback-model'
);

assert.deepStrictEqual(
    helpers.buildOpenRouterReasoningConfig({}, 'janitor', {
        configured: true,
        enabled: true,
        effort: 'xhigh'
    }),
    { enabled: true, exclude: false, effort: 'xhigh' }
);
assert.deepStrictEqual(
    helpers.buildOpenRouterReasoningConfig({}, 'default', {
        configured: true,
        enabled: true,
        budget: 8192
    }),
    { enabled: true, exclude: true, max_tokens: 8192 }
);
assert.deepStrictEqual(
    helpers.buildOpenRouterReasoningConfig({
        reasoning: { effort: 'medium', exclude: false }
    }, 'default', { configured: true, enabled: false }),
    { effort: 'medium', exclude: false },
    'A native OpenRouter reasoning object must be preserved'
);

const root = path.join(__dirname, '..');
const preset = JSON.parse(fs.readFileSync(path.join(root, 'presets', 'frankenstein.json'), 'utf8'));
const compiled = helpers.buildOrderedMessagesFromPreset(
    preset,
    [{ role: 'user', content: 'Begin.' }]
);
const request = helpers.buildOpenRouterRequest({
    model: 'gpt-4',
    messages: [{ role: 'user', content: 'Uncompiled.' }],
    preset_override: 'legacy',
    thinking: true,
    chat_template_kwargs: { enable_thinking: true },
    extra_body: {
        provider: { order: ['Example Provider'] },
        plugins: [{ id: 'response-healing' }]
    }
}, 'anthropic/example-model', compiled, {
    temperature: 0.8,
    max_tokens: 12000
}, true, 'default', {
    configured: false
});

assert.strictEqual(request.model, 'anthropic/example-model');
assert.strictEqual(request.stream, true);
assert.strictEqual(request.temperature, 0.8);
assert.strictEqual(request.max_tokens, 12000);
assert.deepStrictEqual(request.provider, { order: ['Example Provider'] });
assert.deepStrictEqual(request.plugins, [{ id: 'response-healing' }]);
assert.strictEqual(request.extra_body, undefined);
assert.strictEqual(request.preset_override, undefined);
assert.strictEqual(request.chat_template_kwargs, undefined);
assert(request.messages[0].content.includes('<system_state>'));
assert(request.messages[0].content.includes('<internal_states>'));

console.log('openrouter.test.js: all assertions passed');
