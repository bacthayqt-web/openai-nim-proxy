'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const preset = JSON.parse(fs.readFileSync(path.join(root, 'presets', 'frankenstein.json'), 'utf8'));
const janitorOverrides = JSON.parse(fs.readFileSync(path.join(root, 'presets', 'overrides.janitor.json'), 'utf8'));
const helpers = require('../server')._test;

const ids = preset.prompt_order;
const realismId = '019f62e8-892f-700c-9795-53ecb3381d7d';
const freakyId = '019f62e8-892f-700d-b070-4add983aeae1';
const thirdPersonId = '019f62e8-892f-7007-bb78-86fc2ab61efe';
const hybridId = '019f62e8-892f-700a-8e10-84695d624918';
const coloredDialogueId = '019f62e8-892f-7019-be65-715a4949cba0';
const boltId = '634ecfec-1862-4ce0-821e-e31057acadfa';
const maxId = '019f62e8-892f-7032-b004-1869f8bc0782';
const internalStateId = '019f62e8-892f-7027-93ef-159f3d55c410';

assert(ids.includes(realismId), 'Realism Mode must be active');
assert(!ids.includes(freakyId), 'Freaky Mode must be inactive');
assert(ids.includes(thirdPersonId), 'Third-person POV must be active');
assert(!ids.includes(hybridId), 'Hybrid POV must be inactive');
assert(ids.includes(boltId), 'BOLT reasoning must be active');
assert(!ids.includes(maxId), 'MAX reasoning must be inactive');
assert(!ids.includes(coloredDialogueId), 'Colored Dialogue must be inactive');
assert.strictEqual(preset.profile.nsfw_mode, 'realism');
assert.strictEqual(preset.profile.reasoning, 'bolt');
assert.strictEqual(preset.profile.pov, 'third_person');
assert.strictEqual(preset.profile.colored_dialogue, false);
assert.strictEqual(preset.profile.internal_states.length, 8);
assert.strictEqual(new Set(ids).size, ids.length, 'Prompt order must not contain duplicates');
assert.strictEqual(ids.length, preset.prompts.length, 'Every compiled prompt must be ordered exactly once');

const genericBuilt = helpers.buildOrderedMessagesFromPreset(
    preset,
    [{ role: 'user', content: 'Start.' }]
);
const genericSystem = genericBuilt.find((message) => message.role === 'system');
assert(genericSystem, 'Generic compiled preset must contain a merged system message');
assert(genericSystem.content.includes('<internal_states>'));
assert(genericSystem.content.includes('<!-- GFX_START -->'));
assert(genericSystem.content.includes('<internal_dndsim>'));

const janitorBuilt = helpers.buildOrderedMessagesFromPreset(
    preset,
    [{ role: 'user', content: 'Start.' }],
    janitorOverrides,
    helpers.internalStatePromptIds
);
const janitorSystem = janitorBuilt.find((message) => message.role === 'system');
assert(janitorSystem, 'Janitor compiled preset must contain a merged system message');
assert(janitorSystem.content.includes('# Realism Mode:'));
assert(janitorSystem.content.includes('# Reasoning Rules'));
assert(!janitorSystem.content.includes('# Freaky Mode:'));
assert(!janitorSystem.content.includes('# Reasoning Protocol'));
assert(!janitorSystem.content.includes('<!-- FF5_INTERNAL_STATE'));
assert(!janitorSystem.content.includes('DND SIMULATION LOGIC'));
assert(!janitorSystem.content.includes('Purpose: Persistent relationship engine'));
assert(!janitorSystem.content.includes('worldsimRoll:'));
assert(!janitorSystem.content.includes('Append this entire block as raw HTML'));
assert(janitorSystem.content.includes('Internal States are disabled on this frontend'));
assert(!janitorSystem.content.includes('Ensure internal states created correctly'));
assert(!/<colored_dialogue>|<font\s+color=/i.test(janitorSystem.content));
assert(!/\{\{(?:setvar|getvar|roll)::/.test(janitorSystem.content), 'FF5 macros must be expanded server-side');

const genericHistory = helpers.prepareFF5History([
    { role: 'assistant', content: 'Old.\n<!-- FF5_INTERNAL_STATE\nTURN: 1\nEND_FF5_INTERNAL_STATE -->' },
    { role: 'user', content: 'Next.' },
    { role: 'assistant', content: 'Recent.\n<!-- FF5_INTERNAL_STATE\nTURN: 2\nEND_FF5_INTERNAL_STATE -->' },
    { role: 'user', content: 'Continue.' }
]);
assert(!genericHistory[0].content.includes('FF5_INTERNAL_STATE'), 'Old generic state must be pruned');
assert(genericHistory[2].content.includes('FF5_INTERNAL_STATE'), 'Newest generic state must be retained');

const janitorHistory = helpers.prepareFF5History([
    { role: 'assistant', content: 'Old.\n<!-- FF5_INTERNAL_STATE\nTURN: 1\nEND_FF5_INTERNAL_STATE -->' },
    { role: 'user', content: 'Next.' },
    { role: 'assistant', content: 'Recent.\n### INTERNAL STATES\n[WORLD SIM]\nEvent' },
    { role: 'user', content: 'Continue.' }
], true);
assert.strictEqual(janitorHistory[0].content, 'Old.', 'Janitor hidden state must be removed from history');
assert.strictEqual(janitorHistory[2].content, 'Recent.', 'Janitor visible state must be removed from history');

assert.strictEqual(helpers.detectFrontend({ path: '/janitor/v1/chat/completions' }), 'janitor');
assert.strictEqual(helpers.detectFrontend({ path: '/v1/chat/completions' }), 'default');

console.log('preset.test.js: all assertions passed');
