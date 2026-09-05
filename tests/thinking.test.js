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

const misplacedStateThenNarrative = [
    '### INTERNAL STATES',
    '',
    '**TURN:** 2',
    '',
    '#### WORLD SIM',
    '- Event: rumor spreads',
    '',
    '#### PHYSICS, ENGINE & WORLD',
    '- **Environment:** Quiet pharmacy',
    '- **Physics:** Jane and Antoine are one meter apart',
    '',
    '[ 🕰️ 10:19 AM | 🗓️ Day 1 | 📍 Pharmacy | ☀️ Clear ]',
    '',
    '*Jane sets down her mug.*',
    '',
    '"You got tall."'
].join('\n');

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
const kimiK3 = config('moonshotai/kimi-k3', {}, true);
assert.strictEqual(kimiK3.chat_template_kwargs, undefined);
assert.deepStrictEqual(kimiK3.top_level, { reasoning_effort: 'high' });

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

let janitorStateJson = null;
helpers.handleNonStream({
    choices: [{
        index: 0,
        message: {
            role: 'assistant',
            reasoning: 'Private analysis.\n\n### INTERNAL STATES\n[WORLD SIM]\nEvent',
            content: 'Final answer.'
        },
        finish_reason: 'stop'
    }]
}, 'gpt-4-0613', {
    json(value) { janitorStateJson = value; },
    status() { return this; }
}, 'janitor', true);
const mergedStateContent = janitorStateJson.choices[0].message.content;
const mergedStateClose = mergedStateContent.indexOf('</think>');
assert.strictEqual((mergedStateContent.match(/<think>/g) || []).length, 1);
assert.strictEqual(mergedStateContent.indexOf('<think>'), 0);
assert(mergedStateContent.indexOf('Private analysis.') < mergedStateClose);
assert(mergedStateContent.indexOf('### INTERNAL STATES') < mergedStateClose);
assert(mergedStateContent.indexOf('Final answer.') > mergedStateClose);

let relocatedStateJson = null;
helpers.handleNonStream({
    choices: [{
        index: 0,
        message: {
            role: 'assistant',
            content: '<think>0. Audit.\n\nInternal states then narrative.\n</think>\n\n' + misplacedStateThenNarrative
        },
        finish_reason: 'stop'
    }]
}, 'gpt-4-0613', {
    json(value) { relocatedStateJson = value; },
    status() { return this; }
}, 'janitor', true);
const relocatedStateContent = relocatedStateJson.choices[0].message.content;
const relocatedStateClose = relocatedStateContent.indexOf('</think>');
assert.strictEqual((relocatedStateContent.match(/### INTERNAL STATES/g) || []).length, 1);
assert(relocatedStateContent.indexOf('### INTERNAL STATES') < relocatedStateClose);
assert(relocatedStateContent.indexOf('[ 🕰️ 10:19 AM') > relocatedStateClose);
assert(relocatedStateContent.indexOf('*Jane sets down her mug.*') > relocatedStateClose);

let deduplicatedStateJson = null;
helpers.handleNonStream({
    choices: [{
        index: 0,
        message: {
            role: 'assistant',
            reasoning: 'Audit.\n\n### INTERNAL STATES\n\n**TURN:** 2\n\n#### WORLD SIM\n- Event: rumor spreads',
            content: misplacedStateThenNarrative
        },
        finish_reason: 'stop'
    }]
}, 'gpt-4-0613', {
    json(value) { deduplicatedStateJson = value; },
    status() { return this; }
}, 'janitor', true);
const deduplicatedStateContent = deduplicatedStateJson.choices[0].message.content;
assert.strictEqual((deduplicatedStateContent.match(/### INTERNAL STATES/g) || []).length, 1);
assert(deduplicatedStateContent.indexOf('[ 🕰️ 10:19 AM') > deduplicatedStateContent.indexOf('</think>'));

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

async function testStreamedReasoningWithState() {
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

    helpers.handleStream(input, res, 'janitor', true);
    input.write(
        'data: ' + JSON.stringify({
            choices: [{
                delta: {
                    reasoning: 'Stream trace.\n\n### INTERNAL STATES\n[WORLD SIM]\nEvent'
                }
            }]
        }) + '\n\n'
    );
    await new Promise((resolve) => setImmediate(resolve));
    assert(
        writes.join('').includes('<think>\\nStream trace.'),
        'Janitor reasoning must be forwarded before the upstream response completes'
    );

    input.end(
        'data: ' + JSON.stringify({ choices: [{ delta: { content: 'Stream answer.' } }] }) + '\n\n' +
        'data: [DONE]\n\n'
    );
    await ended;

    let content = '';
    writes.join('').split(/\n\n/).forEach(function(event) {
        if (event.indexOf('data: ') !== 0 || event === 'data: [DONE]') return;
        const parsed = JSON.parse(event.slice(6));
        const delta = parsed.choices && parsed.choices[0] && parsed.choices[0].delta;
        if (delta && typeof delta.content === 'string') content += delta.content;
    });

    const closeAt = content.indexOf('</think>');
    assert.strictEqual(content.indexOf('<think>'), 0);
    assert.strictEqual((content.match(/<think>/g) || []).length, 1);
    assert(content.indexOf('Stream trace.') < closeAt);
    assert(content.indexOf('### INTERNAL STATES') < closeAt);
    assert(content.indexOf('Stream answer.') > closeAt);
}

async function testStreamedReasoningRelocatesVisibleStatePrefix() {
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

    helpers.handleStream(input, res, 'janitor', true);
    input.write(
        'data: ' + JSON.stringify({
            choices: [{ delta: { reasoning: '0. Audit.\n\nInternal states then narrative.' } }]
        }) + '\n\n'
    );
    await new Promise((resolve) => setImmediate(resolve));
    assert(writes.join('').includes('<think>\\n0. Audit.'));

    const splitAt = misplacedStateThenNarrative.indexOf('#### PHYSICS');
    input.end(
        'data: ' + JSON.stringify({ choices: [{ delta: { content: misplacedStateThenNarrative.slice(0, 13) } }] }) + '\n\n' +
        'data: ' + JSON.stringify({ choices: [{ delta: { content: misplacedStateThenNarrative.slice(13, splitAt) } }] }) + '\n\n' +
        'data: ' + JSON.stringify({ choices: [{ delta: { content: misplacedStateThenNarrative.slice(splitAt) } }] }) + '\n\n' +
        'data: [DONE]\n\n'
    );
    await ended;

    let content = '';
    writes.join('').split(/\n\n/).forEach(function(event) {
        if (event.indexOf('data: ') !== 0 || event === 'data: [DONE]') return;
        const parsed = JSON.parse(event.slice(6));
        const delta = parsed.choices && parsed.choices[0] && parsed.choices[0].delta;
        if (delta && typeof delta.content === 'string') content += delta.content;
    });

    const closeAt = content.indexOf('</think>');
    assert.strictEqual((content.match(/### INTERNAL STATES/g) || []).length, 1);
    assert(content.indexOf('### INTERNAL STATES') < closeAt);
    assert(content.indexOf('[ 🕰️ 10:19 AM') > closeAt);
    assert(content.indexOf('*Jane sets down her mug.*') > closeAt);
}

async function testLiteralLeadingStateThinkStreams() {
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

    helpers.handleStream(input, res, 'janitor', true);
    input.write(
        'data: ' + JSON.stringify({ choices: [{ delta: { content: '<thi' } }] }) + '\n\n' +
        'data: ' + JSON.stringify({
            choices: [{
                delta: {
                    content: 'nk>\n### INTERNAL STATES\n[WORLD SIM]\nEvent'
                }
            }]
        }) + '\n\n'
    );
    await new Promise((resolve) => setImmediate(resolve));
    assert(
        writes.join('').includes('<think>\\n### INTERNAL STATES'),
        'A literal leading think block must stream before completion'
    );

    input.end(
        'data: ' + JSON.stringify({
            choices: [{ delta: { content: '\n</think>\n\nLiteral answer.' } }]
        }) + '\n\n' +
        'data: [DONE]\n\n'
    );
    await ended;

    let content = '';
    writes.join('').split(/\n\n/).forEach(function(event) {
        if (event.indexOf('data: ') !== 0 || event === 'data: [DONE]') return;
        const parsed = JSON.parse(event.slice(6));
        const delta = parsed.choices && parsed.choices[0] && parsed.choices[0].delta;
        if (delta && typeof delta.content === 'string') content += delta.content;
    });

    const closeAt = content.indexOf('</think>');
    assert.strictEqual(content.indexOf('<think>'), 0);
    assert(content.indexOf('### INTERNAL STATES') < closeAt);
    assert(content.indexOf('Literal answer.') > closeAt);
}

async function testLiteralThinkRelocatesPostThinkState() {
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

    helpers.handleStream(input, res, 'janitor', true);
    input.write(
        'data: ' + JSON.stringify({
            choices: [{ delta: { content: '<think>0. Audit.\n\nInternal states then narrative.\n</think>\n\n### INTER' } }]
        }) + '\n\n'
    );
    await new Promise((resolve) => setImmediate(resolve));
    assert(writes.join('').includes('<think>0. Audit.'));
    assert(!writes.join('').includes('</think>'), 'The close tag must wait for state-prefix inspection');

    input.end(
        'data: ' + JSON.stringify({
            choices: [{ delta: { content: misplacedStateThenNarrative.slice('### INTER'.length) } }]
        }) + '\n\n' +
        'data: [DONE]\n\n'
    );
    await ended;

    let content = '';
    writes.join('').split(/\n\n/).forEach(function(event) {
        if (event.indexOf('data: ') !== 0 || event === 'data: [DONE]') return;
        const parsed = JSON.parse(event.slice(6));
        const delta = parsed.choices && parsed.choices[0] && parsed.choices[0].delta;
        if (delta && typeof delta.content === 'string') content += delta.content;
    });

    const closeAt = content.indexOf('</think>');
    assert.strictEqual((content.match(/### INTERNAL STATES/g) || []).length, 1);
    assert(content.indexOf('### INTERNAL STATES') < closeAt);
    assert(content.indexOf('[ 🕰️ 10:19 AM') > closeAt);
    assert(content.indexOf('*Jane sets down her mug.*') > closeAt);
}

Promise.all([
    testStreamedReasoningAlias(),
    testStreamedReasoningWithState(),
    testStreamedReasoningRelocatesVisibleStatePrefix(),
    testLiteralLeadingStateThinkStreams(),
    testLiteralThinkRelocatesPostThinkState()
]).then(function() {
    console.log('thinking.test.js: all assertions passed');
}).catch(function(error) {
    console.error(error.stack || error);
    process.exitCode = 1;
});
