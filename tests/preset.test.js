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

const hiddenStateOverride = janitorOverrides[internalStateId];
assert(hiddenStateOverride, 'Janitor Internal States override is missing');
assert(hiddenStateOverride.includes('<!-- FF5_INTERNAL_STATE'));
assert(hiddenStateOverride.includes('END_FF5_INTERNAL_STATE -->'));
assert(hiddenStateOverride.includes('{{getvar::invTemplate}}'));
assert(hiddenStateOverride.includes('{{getvar::worldsimTemplate}}'));

const built = helpers.buildOrderedMessagesFromPreset(
    preset,
    [{ role: 'user', content: 'Start.' }],
    janitorOverrides
);
const system = built.find((message) => message.role === 'system');
assert(system, 'Compiled preset must contain a merged system message');
assert(system.content.includes('<!-- FF5_INTERNAL_STATE'));
assert(system.content.includes('# Realism Mode:'));
assert(system.content.includes('# Reasoning Rules'));
assert(!system.content.includes('# Freaky Mode:'));
assert(!system.content.includes('# Reasoning Protocol'));
assert(!/<colored_dialogue>|<font\s+color=/i.test(system.content));
assert(!/\{\{(?:setvar|getvar|roll)::/.test(system.content), 'FF5 macros must be expanded server-side');

const history = helpers.prepareFF5History([
    { role: 'assistant', content: 'Old.\n<!-- FF5_INTERNAL_STATE\nTURN: 1\nEND_FF5_INTERNAL_STATE -->' },
    { role: 'user', content: 'Next.' },
    { role: 'assistant', content: 'Recent.\n<!-- FF5_INTERNAL_STATE\nTURN: 2\nEND_FF5_INTERNAL_STATE -->' },
    { role: 'user', content: 'Continue.' }
]);
assert(!history[0].content.includes('FF5_INTERNAL_STATE'), 'Old hidden state must be pruned');
assert(history[2].content.includes('FF5_INTERNAL_STATE'), 'Newest hidden state must be retained');

assert.strictEqual(helpers.detectFrontend({ path: '/janitor/v1/chat/completions' }), 'janitor');
assert.strictEqual(helpers.detectFrontend({ path: '/v1/chat/completions' }), 'default');

console.log('preset.test.js: all assertions passed');
