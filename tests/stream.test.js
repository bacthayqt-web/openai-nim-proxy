'use strict';

const assert = require('assert');
const { PassThrough } = require('stream');
const helpers = require('../server')._test;

function assertNoInternalState(text) {
    assert(!/FF5_INTERNAL_STATE/i.test(text), 'Janitor output must not contain a hidden state comment');
    assert(!/<\/?internal[_\s-]*states?/i.test(text), 'Janitor output must not contain state XML');
    assert(!/(?:^|\n)\s*(?:#{1,6}\s*)?(?:🎬\s*)?INTERNAL STATES?/im.test(text), 'Janitor output must not contain a state heading');
    assert(!/(?:^|\n)\s*\[(?:NPC AGENDAS|WORLD SIM|GM NOTEBOOK|DND TASK SIM|INTERNAL THOUGHTS)\]/im.test(text), 'Janitor output must not contain state sections');
}

function assertLeadingStateBox(text, narrativeText, stateText) {
    const openAt = text.indexOf('<think>');
    const closeAt = text.indexOf('</think>');
    const narrativeAt = text.indexOf(narrativeText);
    const stateAt = text.indexOf(stateText);

    assert.strictEqual(openAt, 0, 'Janitor think box must lead the message');
    assert(closeAt > openAt, 'Janitor think box must close');
    assert(stateAt > openAt && stateAt < closeAt, 'Internal States must be inside the leading think box');
    assert(narrativeAt > closeAt, 'Visible narrative must follow the think box');
    assert.strictEqual((text.match(/<think>/g) || []).length, 1, 'Janitor output must contain one think box');
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

async function runFrontendStream(contents, frontend, transportCuts) {
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

    helpers.handleStream(input, res, frontend, true);
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
    const liveInput = new PassThrough();
    const liveWrites = [];
    let resolveLiveEnd;
    const liveEnded = new Promise((resolve) => { resolveLiveEnd = resolve; });
    const liveRes = {
        headersSent: false,
        setHeader() {},
        write(chunk) { liveWrites.push(String(chunk)); },
        end() { resolveLiveEnd(); }
    };
    helpers.handleStream(liveInput, liveRes, 'janitor', true);
    liveInput.write(makeEvent('N'.repeat(600)));
    await new Promise((resolve) => setImmediate(resolve));
    assert(
        parseOutput(liveWrites).length > 0,
        'Janitor narrative must begin streaming before the upstream response ends'
    );
    liveInput.end('data: [DONE]\n\n');
    await liveEnded;

    const visibleMarkdown = helpers.hideJanitorInternalState(
        'Narrative.\n\n### INTERNAL STATES\n[GM NOTEBOOK]\nSecret -- unfinished'
    );
    assert.strictEqual(visibleMarkdown, 'Narrative.');
    assertNoInternalState(visibleMarkdown);

    const genericHtml = helpers.hideJanitorInternalState(
        'Narrative.\n<!-- GFX_START -->\n<internal_states><details><summary>🎬 INTERNAL STATES</summary>Secret</details></internal_states>\n<!-- GFX_END -->'
    );
    assert.strictEqual(genericHtml, 'Narrative.');
    assertNoInternalState(genericHtml);

    const malformed = helpers.hideJanitorInternalState(
        'Narrative.\n<!-- FF5_INTERNAL_STATE\nTURN: 4\n[WORLD SIM]\nEvent'
    );
    assert.strictEqual(malformed, 'Narrative.');
    assertNoInternalState(malformed);

    const alreadyHidden = helpers.hideJanitorInternalState(
        'Narrative.\n<!-- FF5_INTERNAL_STATE\nTURN: 5\n[QUESTS]\nNone\nEND_FF5_INTERNAL_STATE -->'
    );
    assert.strictEqual(alreadyHidden, 'Narrative.');
    assertNoInternalState(alreadyHidden);

    const xml = helpers.hideJanitorInternalState(
        'Narrative.\n<internal_state>[BONDS]\nNPC: 3</internal_state>'
    );
    assert.strictEqual(xml, 'Narrative.');
    assertNoInternalState(xml);

    const orphan = helpers.hideJanitorInternalState(
        'Narrative.\n\n[INTERNAL THOUGHTS]\nNPC: leave now'
    );
    assert.strictEqual(orphan, 'Narrative.');
    assertNoInternalState(orphan);

    const popIn = '<!-- GFX_START --><div>📱 Phone message</div><!-- GFX_END -->';
    assert.strictEqual(helpers.hideJanitorInternalState('Narrative.\n' + popIn), 'Narrative.\n' + popIn);

    const progressive = helpers.createJanitorStateStream();
    const longNarrative = 'N'.repeat(600);
    assert(progressive.push(longNarrative).length > 0, 'Long narrative must stream before completion');
    assert(progressive.finish().length > 0, 'Buffered narrative tail must flush at completion');

    const stripping = helpers.createJanitorStateStream();
    assert.strictEqual(stripping.push('Narrative.\n\n### INTERNAL STATES\n[QUESTS]').trim(), 'Narrative.');
    assert.strictEqual(stripping.push('\nSecret'), '');
    const wrappedState = stripping.finish();
    assert(wrappedState.includes('<!-- FF5_INTERNAL_STATE'), 'A misplaced Janitor state tail must fall back to a hidden comment');
    assert(!wrappedState.includes('<think>'), 'A trailing state must not create a broken end-of-message think box');
    assert(wrappedState.includes('#### QUESTS'));

    const streamedMarkdown = await runFrontendStream([
        'Narrative paragraph.',
        '\n\n### INTER',
        'NAL STATES\n[GM NOTEBOOK]\nSecret -- note'
    ], 'janitor', [1, 2, 5, 3, 13, 8, 21]);
    assert(streamedMarkdown.content.includes('Narrative paragraph.'));
    assert(streamedMarkdown.content.includes('<!-- FF5_INTERNAL_STATE'));
    assert(!streamedMarkdown.content.includes('<think>'));
    assert(streamedMarkdown.content.includes('#### GM NOTEBOOK'));

    const streamedHtml = await runFrontendStream([
        'Narrative paragraph.\n',
        '<!-- GFX_',
        'START -->\n<internal_',
        'states><details><summary>INTERNAL STATES</summary>Secret</details></internal_states><!-- GFX_END -->'
    ], 'janitor', [7, 1, 19, 4, 2, 33]);
    assert(streamedHtml.content.includes('Narrative paragraph.'));
    assert(streamedHtml.content.includes('<!-- FF5_INTERNAL_STATE'));
    assert(!streamedHtml.content.includes('<think>'));
    assert(streamedHtml.content.includes('### INTERNAL STATES'));

    const streamedPopIn = await runFrontendStream([
        'Narrative.\n',
        '<!-- GFX_START --><div>📱 Phone message</div><!-- GFX_END -->'
    ], 'janitor', [3, 9, 2, 17]);
    assert(streamedPopIn.content.includes('📱 Phone message'), 'Ordinary Pop-in Graphics must remain visible');
    assertNoInternalState(streamedPopIn.content);

    const genericHtmlStream = await runFrontendStream([
        'Narrative paragraph.\n',
        '<!-- GFX_START -->\n<internal_states><details><summary>🎬 INTERNAL STATES</summary>',
        '<details><summary>WORLD SIM</summary>Event</details></details></internal_states><!-- GFX_END -->'
    ], 'default', [5, 2, 17, 3, 29]);
    assert(genericHtmlStream.content.includes('INTERNAL STATES'), 'Generic state heading must remain visible');
    assert(genericHtmlStream.content.includes('<details style='), 'Generic native HTML must receive FF5 styling');
    assert(genericHtmlStream.content.includes('WORLD SIM'), 'Generic state sections must remain present');

    const genericFallbackStream = await runFrontendStream([
        'Narrative paragraph.\n',
        '<!-- FF5_INTERNAL_STATE\nTURN: 7\n[WORLD SIM]\nEvent\nEND_FF5_INTERNAL_STATE -->'
    ], 'default', [4, 11, 1, 23]);
    assert(genericFallbackStream.content.includes('<details style='), 'Generic hidden-comment variant must become a visible panel');
    assert(/Turn:\s*7/i.test(genericFallbackStream.content));
    assert(genericFallbackStream.content.includes('WORLD SIM'));

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
    assert(nonStreamContent.includes('Narrative.'));
    assert(nonStreamContent.includes('<think>'));
    assert(nonStreamContent.includes('#### DND TASK SIM'));
    assertLeadingStateBox(nonStreamContent, 'Narrative.', '### INTERNAL STATES');

    let genericNonStreamJson = null;
    helpers.handleNonStream({
        choices: [{
            index: 0,
            message: {
                role: 'assistant',
                content: 'Narrative.\n<!-- FF5_INTERNAL_STATE\nTURN: 8\n[WORLD SIM]\nEvent\nEND_FF5_INTERNAL_STATE -->'
            },
            finish_reason: 'stop'
        }]
    }, 'gpt-4', {
        json(value) { genericNonStreamJson = value; },
        status() { return this; }
    }, 'default', true);
    const genericNonStreamContent = genericNonStreamJson.choices[0].message.content;
    assert(genericNonStreamContent.includes('<details style='));
    assert(/Turn:\s*8/i.test(genericNonStreamContent));
    assert(genericNonStreamContent.includes('WORLD SIM'));

    console.log('stream.test.js: all assertions passed');
})().catch((error) => {
    console.error(error.stack || error);
    process.exitCode = 1;
});
