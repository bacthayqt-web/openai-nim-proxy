'use strict';

const assert = require('assert');
const { PassThrough } = require('stream');
const helpers = require('../server')._test;

function outsideHiddenComment(text) {
    return text.replace(/<!--\s*FF5_INTERNAL_STATE\b[\s\S]*?END_FF5_INTERNAL_STATE\s*-->/g, '');
}

function assertOneValidHiddenState(text) {
    const matches = text.match(/<!--\s*FF5_INTERNAL_STATE\b[\s\S]*?END_FF5_INTERNAL_STATE\s*-->/g) || [];
    assert.strictEqual(matches.length, 1, 'Expected exactly one normalized hidden state comment');
    const body = matches[0]
        .replace(/^<!--\s*FF5_INTERNAL_STATE\s*/, '')
        .replace(/END_FF5_INTERNAL_STATE\s*-->$/, '');
    assert(!body.includes('--'), 'Hidden comment body must not contain a double hyphen');
}

function makeEvent(content) {
    return 'data: ' + JSON.stringify({
        choices: [{ delta: { content } }]
    }) + '\n\n';
}

function parseOutput(writes) {
    let content = '';
    const events = writes.join('').split(/\n\n/);
    for (const event of events) {
        if (!event.startsWith('data: ')) continue;
        const raw = event.slice(6);
        if (!raw || raw === '[DONE]') continue;
        const parsed = JSON.parse(raw);
        const delta = parsed.choices && parsed.choices[0] && parsed.choices[0].delta;
        if (delta && typeof delta.content === 'string') content += delta.content;
    }
    return content;
}

async function runJanitorStream(contents, transportCuts) {
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
    const wire = contents.map(makeEvent).join('') + 'data: [DONE]\n\n';

    if (Array.isArray(transportCuts) && transportCuts.length) {
        let cursor = 0;
        for (const size of transportCuts) {
            if (cursor >= wire.length) break;
            input.write(wire.slice(cursor, cursor + size));
            cursor += size;
        }
        if (cursor < wire.length) input.write(wire.slice(cursor));
        input.end();
    } else {
        input.end(wire);
    }

    await ended;
    return { content: parseOutput(writes), writes };
}

(async function main() {
    const visibleMarkdown = helpers.hideJanitorInternalState(
        'Narrative.\n\n### INTERNAL STATES\n[GM NOTEBOOK]\nSecret -- unfinished'
    );
    assert.strictEqual(outsideHiddenComment(visibleMarkdown).trim(), 'Narrative.');
    assertOneValidHiddenState(visibleMarkdown);

    const genericHtml = helpers.hideJanitorInternalState(
        'Narrative.\n<!-- GFX_START -->\n<internal_states><details><summary>🎬 INTERNAL STATES</summary>Secret</details></internal_states>\n<!-- GFX_END -->'
    );
    assert.strictEqual(outsideHiddenComment(genericHtml).trim(), 'Narrative.');
    assertOneValidHiddenState(genericHtml);
    assert(!outsideHiddenComment(genericHtml).includes('GFX_START'));

    const malformed = helpers.hideJanitorInternalState(
        'Narrative.\n<!-- FF5_INTERNAL_STATE\nTURN: 4\n[WORLD SIM]\nEvent'
    );
    assert.strictEqual(outsideHiddenComment(malformed).trim(), 'Narrative.');
    assertOneValidHiddenState(malformed);

    const alreadyHidden = helpers.hideJanitorInternalState(
        'Narrative.\n<!-- FF5_INTERNAL_STATE\nTURN: 5\n[QUESTS]\nNone\nEND_FF5_INTERNAL_STATE -->'
    );
    assert.strictEqual(outsideHiddenComment(alreadyHidden).trim(), 'Narrative.');
    assertOneValidHiddenState(alreadyHidden);

    const xml = helpers.hideJanitorInternalState(
        'Narrative.\n<internal_state>[BONDS]\nNPC: 3</internal_state>'
    );
    assert.strictEqual(outsideHiddenComment(xml).trim(), 'Narrative.');
    assertOneValidHiddenState(xml);

    const orphan = helpers.hideJanitorInternalState(
        'Narrative.\n\n[INTERNAL THOUGHTS]\nNPC: leave now'
    );
    assert.strictEqual(outsideHiddenComment(orphan).trim(), 'Narrative.');
    assertOneValidHiddenState(orphan);

    const popIn = '<!-- GFX_START --><div>📱 Phone message</div><!-- GFX_END -->';
    assert.strictEqual(helpers.hideJanitorInternalState('Narrative.\n' + popIn), 'Narrative.\n' + popIn);

    const progressive = helpers.createJanitorStateStream();
    const longNarrative = 'N'.repeat(600);
    assert(progressive.push(longNarrative).length > 0, 'Long narrative must stream before completion');
    assert(progressive.finish().length > 0, 'Buffered narrative tail must flush at completion');

    const streamedMarkdown = await runJanitorStream([
        'Narrative paragraph.',
        '\n\n### INTER',
        'NAL STATES\n[GM NOTEBOOK]\nSecret -- note'
    ], [1, 2, 5, 3, 13, 8, 21]);
    assert.strictEqual(outsideHiddenComment(streamedMarkdown.content).trim(), 'Narrative paragraph.');
    assertOneValidHiddenState(streamedMarkdown.content);

    const streamedHtml = await runJanitorStream([
        'Narrative paragraph.\n',
        '<!-- GFX_',
        'START -->\n<internal_',
        'states><details><summary>INTERNAL STATES</summary>Secret</details></internal_states><!-- GFX_END -->'
    ], [7, 1, 19, 4, 2, 33]);
    assert.strictEqual(outsideHiddenComment(streamedHtml.content).trim(), 'Narrative paragraph.');
    assertOneValidHiddenState(streamedHtml.content);

    const streamedPopIn = await runJanitorStream([
        'Narrative.\n',
        '<!-- GFX_START --><div>📱 Phone message</div><!-- GFX_END -->'
    ], [3, 9, 2, 17]);
    assert(streamedPopIn.content.includes('📱 Phone message'), 'Ordinary Pop-in Graphics must remain visible');
    assert(!streamedPopIn.content.includes('FF5_INTERNAL_STATE'));

    let nonStreamJson = null;
    helpers.handleNonStream({
        choices: [{
            index: 0,
            message: {
                role: 'assistant',
                content: 'Narrative.\n\n### INTERNAL STATES\n[DND TASK SIM]\nRoll: 12'
            },
            finish_reason: 'stop'
        }]
    }, 'gpt-4', {
        json(value) { nonStreamJson = value; },
        status() { return this; }
    }, 'janitor', true);
    const nonStreamContent = nonStreamJson.choices[0].message.content;
    assert.strictEqual(outsideHiddenComment(nonStreamContent).trim(), 'Narrative.');
    assertOneValidHiddenState(nonStreamContent);

    console.log('stream.test.js: all assertions passed');
})().catch((error) => {
    console.error(error.stack || error);
    process.exitCode = 1;
});
