'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const preset = JSON.parse(fs.readFileSync(path.join(root, 'presets', 'frankenstein.json'), 'utf8'));
const regexSuite = JSON.parse(fs.readFileSync(path.join(root, 'presets', 'ff5-regex.json'), 'utf8'));
const janitorOverrides = JSON.parse(fs.readFileSync(path.join(root, 'presets', 'overrides.janitor.json'), 'utf8'));
const janitorStateOverrides = JSON.parse(fs.readFileSync(path.join(root, 'presets', 'overrides.janitor-state.json'), 'utf8'));
const janitorJailbreakOverrides = JSON.parse(fs.readFileSync(path.join(root, 'presets', 'overrides.janitor-jailbreak.json'), 'utf8'));
const combinedJanitorOverrides = Object.assign(
    {},
    janitorOverrides,
    janitorStateOverrides,
    janitorJailbreakOverrides
);
const helpers = require('../server')._test;

const ids = preset.prompt_order;
const realismId = '019f62e8-892f-700c-9795-53ecb3381d7d';
const freakyId = '019f62e8-892f-700d-b070-4add983aeae1';
const thirdPersonId = '019f62e8-892f-7007-bb78-86fc2ab61efe';
const hybridId = '019f62e8-892f-700a-8e10-84695d624918';
const coloredDialogueId = '019f62e8-892f-7019-be65-715a4949cba0';
const boltId = '634ecfec-1862-4ce0-821e-e31057acadfa';
const reasoningGateId = 'reasoning_completion_gate';
const maxId = '019f62e8-892f-7032-b004-1869f8bc0782';
const internalStateId = '019f62e8-892f-7027-93ef-159f3d55c410';

assert(ids.includes(realismId), 'Realism Mode must be active');
assert(!ids.includes(freakyId), 'Freaky Mode must be inactive');
assert(ids.includes(thirdPersonId), 'Third-person POV must be active');
assert(!ids.includes(hybridId), 'Hybrid POV must be inactive');
assert(ids.includes(boltId), 'BOLT reasoning must be active');
assert(ids.includes(reasoningGateId), 'The anti-drafting completion gate must be active');
assert(!ids.includes(maxId), 'MAX reasoning must be inactive');
assert(!ids.includes(coloredDialogueId), 'Colored Dialogue must be inactive');
assert.strictEqual(preset.profile.nsfw_mode, 'realism');
assert.strictEqual(preset.profile.preset_version, '5.4');
assert.strictEqual(preset.profile.regex_suite, '3.0');
assert.strictEqual(preset.profile.reasoning, 'bolt');
assert.strictEqual(preset.profile.reasoning_guard, 'single_pass_anti_draft');
assert.strictEqual(preset.profile.reasoning_task_target_words, 30);
assert.strictEqual(preset.profile.pov, 'third_person');
assert.strictEqual(preset.profile.colored_dialogue, false);
assert.strictEqual(preset.profile.npc_voice, 'micro_2.0');
assert.strictEqual(preset.profile.internal_states.length, 8);
assert.strictEqual(new Set(ids).size, ids.length, 'Prompt order must not contain duplicates');
assert.strictEqual(ids.length, preset.prompts.length, 'Every compiled prompt must be ordered exactly once');

const dndPrompt = preset.prompts.find((prompt) => prompt.identifier === '019f62e8-892f-7021-97a6-42e1b83eaad3');
assert(dndPrompt, 'FF5.4 DnD prompt must be present');
assert.strictEqual((dndPrompt.content.match(/<internal_dndsim>/g) || []).length, 1, 'DnD wrapper must open once');
assert.strictEqual((dndPrompt.content.match(/<\/internal_dndsim>/g) || []).length, 1, 'DnD wrapper must close once');

const boltPrompt = preset.prompts.find((prompt) => prompt.identifier === boltId);
const reasoningGate = preset.prompts.find((prompt) => prompt.identifier === reasoningGateId);
assert(boltPrompt.content.includes('Run Tasks 0-10 exactly once and in order'));
assert(boltPrompt.content.includes('No sample dialogue'));
assert(!boltPrompt.content.includes('Brainstorm 3 very distinct'));
assert(!boltPrompt.content.includes('write dialogue examples'));
assert(reasoningGate.content.includes('STOP CONDITION'));
assert(reasoningGate.content.includes('never as a composition workspace'));
assert(reasoningGate.content.includes('Never write a placeholder'));
assert(reasoningGate.content.includes('`</think>` followed by `### INTERNAL STATES` is forbidden'));
assert.strictEqual(ids[ids.length - 1], reasoningGateId, 'The completion gate must be the final preset instruction');

const genericBuilt = helpers.buildOrderedMessagesFromPreset(
    preset,
    [{ role: 'user', content: 'Start.' }]
);
const genericSystem = genericBuilt.find((message) => message.role === 'system');
assert(genericSystem, 'Generic compiled preset must contain a merged system message');
assert(genericSystem.content.includes('<internal_states>'));
assert(genericSystem.content.includes('<!-- GFX_START -->'));
assert(genericSystem.content.includes('<internal_dndsim>'));
assert(genericSystem.content.includes('# BOLT Completion Gate'));

const janitorBuilt = helpers.buildOrderedMessagesFromPreset(
    preset,
    [{ role: 'user', content: 'Start.' }],
    combinedJanitorOverrides,
    helpers.internalStatePromptIds
);
const janitorSystem = janitorBuilt.find((message) => message.role === 'system');
assert(janitorSystem, 'Janitor compiled preset must contain a merged system message');
assert(janitorSystem.content.includes('# Realism Mode:'));
assert(janitorSystem.content.includes('# BOLT Reasoning Checklist'));
assert(janitorSystem.content.includes('# BOLT Completion Gate'));
assert(!janitorSystem.content.includes('# Freaky Mode:'));
assert(!janitorSystem.content.includes('# Reasoning Protocol'));
assert(!janitorSystem.content.includes('<!-- FF5_INTERNAL_STATE'));
assert(!janitorSystem.content.includes('DND SIMULATION LOGIC'));
assert(!janitorSystem.content.includes('Purpose: Persistent relationship engine'));
assert(!janitorSystem.content.includes('worldsimRoll:'));
assert(!janitorSystem.content.includes('Append this entire block as raw HTML'));
assert(janitorSystem.content.includes('Internal States are disabled on this frontend'));
assert(!janitorSystem.content.includes('Ensure internal states created correctly'));
assert(!/<colored_dialogue>/i.test(janitorSystem.content));
assert(!/<font\s+color=/i.test(janitorSystem.content));
assert(!/\{\{(?:setvar|getvar|roll)::/.test(janitorSystem.content), 'FF5 macros must be expanded server-side');

const janitorStateBuilt = helpers.buildOrderedMessagesFromPreset(
    preset,
    [{ role: 'user', content: 'Start.' }],
    combinedJanitorOverrides,
    []
);
const janitorStateSystem = janitorStateBuilt.find((message) => message.role === 'system');
assert(janitorStateSystem.content.includes('final part of private reasoning'));
assert(janitorStateSystem.content.includes('Never append Internal States after the narrative'));
assert(janitorStateSystem.content.includes('DO NOT write a placeholder'));
assert(janitorStateSystem.content.includes('FORBIDDEN OUTPUT ORDER'));
assert(janitorStateSystem.content.includes('DND SIM — Dice=law'));
assert(janitorStateSystem.content.includes('#### PHYSICS, ENGINE & WORLD'));
assert(!janitorStateSystem.content.includes('<details>'));
assert(!janitorStateSystem.content.includes('<!-- FF5_INTERNAL_STATE'));
assert(!janitorStateSystem.content.includes('created correctly at end of every response'));

const genericHistory = helpers.prepareFF5History([
    { role: 'assistant', content: 'Old.\n<!-- FF5_INTERNAL_STATE\nTURN: 1\nEND_FF5_INTERNAL_STATE -->' },
    { role: 'user', content: 'Next.' },
    { role: 'assistant', content: 'Recent.\n<!-- FF5_INTERNAL_STATE\nTURN: 2\nEND_FF5_INTERNAL_STATE -->' },
    { role: 'user', content: 'Continue.' }
]);
assert(!genericHistory[0].content.includes('FF5_INTERNAL_STATE'), 'Old generic state must be pruned');
assert(genericHistory[2].content.includes('<internal_states>'), 'Newest generic state must be retained');
assert(genericHistory[2].content.includes('TURN: 2'), 'Newest generic state content must survive normalization');

const janitorHistory = helpers.prepareFF5History([
    { role: 'assistant', content: 'Old.\n<!-- FF5_INTERNAL_STATE\nTURN: 1\nEND_FF5_INTERNAL_STATE -->' },
    { role: 'user', content: 'Next.' },
    { role: 'assistant', content: 'Recent.\n### INTERNAL STATES\n[WORLD SIM]\nEvent' },
    { role: 'user', content: 'Continue.' }
], true);
assert.strictEqual(janitorHistory[0].content, 'Old.', 'Janitor hidden state must be removed from history');
assert.strictEqual(janitorHistory[2].content, 'Recent.', 'Janitor visible state must be removed from history');

const reorderedJanitorHistory = helpers.prepareFF5History([
    {
        role: 'assistant',
        content: '<think>\n### INTERNAL STATES\n\n#### WORLD SIM\nOld event\n</think>\n\nOld narrative.'
    },
    { role: 'user', content: 'Next.' },
    {
        role: 'assistant',
        content: '<think>\nNative reasoning.\n\n### INTERNAL STATES\n\n#### WORLD SIM\nRecent event\n</think>\n\nRecent narrative.'
    },
    { role: 'user', content: 'Continue.' }
], false, 'janitor');
assert.strictEqual(
    reorderedJanitorHistory[0].content,
    'Old narrative.',
    'Pruning an old leading state box must preserve its visible narrative'
);
assert(!reorderedJanitorHistory[2].content.includes('Native reasoning.'));
assert(!reorderedJanitorHistory[2].content.includes('<think>'));
assert(reorderedJanitorHistory[2].content.includes('Recent narrative.'));
assert(reorderedJanitorHistory[2].content.includes('<internal_states>'));
assert(reorderedJanitorHistory[2].content.includes('Recent event'));
assert(
    reorderedJanitorHistory[2].content.indexOf('Recent narrative.') <
        reorderedJanitorHistory[2].content.indexOf('<internal_states>'),
    'The restored semantic state must follow the narrative in model context'
);

assert.strictEqual(helpers.detectFrontend({ path: '/janitor/v1/chat/completions' }), 'janitor');
assert.strictEqual(helpers.detectFrontend({ path: '/v1/chat/completions' }), 'default');

assert.strictEqual(regexSuite.length, 25, 'The complete Regex 3.0 suite must be installed');
assert(regexSuite.some((script) => script.scriptName === 'FF5 Delete - Untagged Thoughts'));
assert(regexSuite.some((script) => script.scriptName === 'FF5 Repair - GFX Unfence'));
assert(regexSuite.some((script) => script.scriptName === 'FF5 - Universal Dialogue Colorizer'));
assert(regexSuite.some((script) => script.scriptName === 'FF5 - Context Saver (Universal)'));
regexSuite.forEach((script) => {
    const lastSlash = script.findRegex.lastIndexOf('/');
    assert(lastSlash > 0, `${script.scriptName} must contain a regex literal`);
    assert.doesNotThrow(
        () => new RegExp(script.findRegex.slice(1, lastSlash), script.findRegex.slice(lastSlash + 1)),
        `${script.scriptName} must compile in Node.js`
    );
});

console.log('preset.test.js: all assertions passed');
