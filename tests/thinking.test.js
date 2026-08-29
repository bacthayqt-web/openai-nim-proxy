'use strict';

process.env.SHOW_REASONING = 'true';
process.env.ENABLE_THINKING_MODE = 'true';
process.env.REASONING_EFFORT = 'high';

const assert = require('assert');
const { PassThrough } = require('stream');
const helpers = require('../server')._test;

function config(model, body, enabled, effort, budget) {
    return helpers.buildThinkingConfig(model, body || {}, {
        enabled,
        effort: effort || 'high',
        budget
    });
}

assert.deepStrictEqual(
    config('deepseek-ai/deepseek-v4-flash-0731', {}, true).chat_template_kwargs,
    { thinking: true, reasoning_effort: 'high' }
);
assert.deepStrictEqual(
    config('deepseek-ai/deepseek-v4-pro', {}, false).chat_template_kwargs,
    { thinking: false }
);
assert.strictEqual(
    config('deepseek-ai/deepseek-r1', {}, true).chat_template_kwargs,
    undefined,
    'Native DeepSeek reasoners must not receive V4-only flags'
);
assert.deepStrictEqual(
    config('z-ai/glm-5.2', {}, true).chat_template_kwargs,
    { enable_thinking: true }
);
const kimiK3 = config('moonshotai/kimi-k3', {}, false, 'max');
assert.strictEqual(kimiK3.chat_template_kwargs, undefined);
assert.deepStrictEqual(kimiK3.top_level, { reasoning_effort: 'max' });
assert.strictEqual(kimiK3.enabled, true, 'Kimi K3 reasoning is always enabled');
assert.strictEqual(kimiK3.effort, 'max');

const kimiK3SdkStyle = config('moonshotai/kimi-k3', {
    extra_body: { reasoning_effort: 'medium' }
}, true);
assert.deepStrictEqual(kimiK3SdkStyle.top_level, { reasoning_effort: 'high' });

const qwen = config('qwen/qwen3.8', {}, true, 'medium');
assert.deepStrictEqual(qwen.chat_template_kwargs, { enable_thinking: true });
assert.deepStrictEqual(qwen.top_level, { reasoning_effort: 'medium' });

const nemotron = config('nvidia/nemotron-3-ultra-550b-a55b', {}, true, 'high', 16384);
assert.deepStrictEqual(nemotron.chat_template_kwargs, { enable_thinking: true });
assert.deepStrictEqual(nemotron.top_level, { reasoning_budget: 16384 });

assert.deepStrictEqual(
    config('thinkingmachines/inkling', {}, false).chat_template_kwargs,
    { reasoning_effort: 'none' }
);
assert.strictEqual(
    config('minimaxai/minimax-m3', {}, true).chat_template_kwargs,
    undefined,
    'Native-reasoning models must use provider defaults unless a client supplies options'
);

const sdkStyle = config('deepseek-ai/deepseek-v4-flash-0731', {
    extra_body: {
        chat_template_kwargs: { thinking: true, reasoning_effort: 'max' }
    }
}, false);
assert.deepStrictEqual(sdkStyle.chat_template_kwargs, {
    thinking: true,
    reasoning_effort: 'max'
});

assert.strictEqual(
    helpers.extractReasoning({ reasoning: 'reasoning field' }),
    'reasoning field'
);
assert.strictEqual(
    helpers.extractReasoning({
        reasoning_details: [
            { type: 'reasoning.text', text: 'first' },
            { type: 'reasoning.text', text: 'second' }
        ]
    }),
    'first\nsecond'
);

let janitorJson = null;
helpers.handleNonStream({
    choices: [{
        index: 0,
        message: {
            role: 'assistant',
            reasoning: 'Private analysis.',
            content: 'Final answer.'
        },
        finish_reason: 'stop'
    }]
}, 'gpt-4-0613', {
    json(value) { janitorJson = value; },
    status() { return this; }
}, 'janitor', false);
assert.strictEqual(
    janitorJson.choices[0].message.content,
    '<think>\nPrivate analysis.\n</think>\n\nFinal answer.'
);

let genericJson = null;
helpers.handleNonStream({
    choices: [{
        message: {
            content: '<analysis>Hidden literal trace.</analysis>Visible answer.'
        }
    }]
}, 'gpt-4-0613', {
    json(value) { genericJson = value; },
    status() { return this; }
}, 'default', false);
assert.strictEqual(genericJson.choices[0].message.content, 'Visible answer.');

async function testStreamedReasoningAlias() {
    const input = new PassThrough();
    const writes = [];
    let resolveEnd;
    const ended = new Promise((resolve) => { resolveEnd = resolve; });
    const res = {
        headersSent: false,
        setHeader() {},
        write(chunk) { writes.push(String(chunk)); },
        end() { resolveEnd(); }
    };

    helpers.handleStream(input, res, 'janitor', false);
    input.end(
        'data: ' + JSON.stringify({ choices: [{ delta: { reasoning: 'Stream trace.' } }] }) + '\n\n' +
        'data: ' + JSON.stringify({ choices: [{ delta: { content: 'Stream answer.' } }] }) + '\n\n' +
        'data: [DONE]\n\n'
    );
    await ended;

    const wire = writes.join('');
    assert(wire.includes('<think>\\nStream trace.'));
    assert(wire.includes('</think>\\n\\nStream answer.'));
    assert(!wire.includes('"reasoning"'));
}

testStreamedReasoningAlias().then(function() {
    console.log('thinking.test.js: all assertions passed');
}).catch(function(error) {
    console.error(error.stack || error);
    process.exitCode = 1;
});
