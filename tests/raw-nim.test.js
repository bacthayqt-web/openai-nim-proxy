'use strict';

process.env.NIM_API_KEY = 'test-key';

const assert = require('assert');
const axios = require('axios');

const originalPost = axios.post;
let capturedUrl;
let capturedRequest;

axios.post = async function(url, request) {
    capturedUrl = url;
    capturedRequest = request;
    return {
        status: 200,
        data: {
            id: 'chatcmpl-raw-test',
            object: 'chat.completion',
            created: 1,
            model: request.model,
            choices: [{
                index: 0,
                message: { role: 'assistant', content: 'Raw response.' },
                finish_reason: 'stop'
            }],
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 }
        }
    };
};

const app = require('../server');
const helpers = app._test;

assert.strictEqual(
    helpers.isRawNimRoute({ path: '/raw/v1/chat/completions' }),
    true
);
assert.strictEqual(
    helpers.isRawNimRoute({ path: '/v1/chat/completions' }),
    false
);

const messages = [
    { role: 'system', content: 'Frontend system prompt.' },
    { role: 'user', content: 'Hello from the frontend.' },
    { role: 'system', content: 'Keep this message in this exact position.' }
];

const server = app.listen(0, '127.0.0.1', async function() {
    try {
        const address = server.address();
        const response = await fetch(
            'http://127.0.0.1:' + address.port + '/raw/v1/chat/completions',
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: 'nvidia/test-model',
                    messages,
                    temperature: 0.4,
                    max_tokens: 512,
                    stream: false
                })
            }
        );
        const body = await response.json();

        assert.strictEqual(response.status, 200);
        assert.strictEqual(capturedUrl, 'https://integrate.api.nvidia.com/v1/chat/completions');
        assert.deepStrictEqual(
            capturedRequest.messages,
            messages,
            'The raw NIM route must not inject, merge, remove, or reorder prompt messages'
        );
        assert.strictEqual(capturedRequest.model, 'nvidia/test-model');
        assert.strictEqual(capturedRequest.temperature, 0.4);
        assert.strictEqual(capturedRequest.max_tokens, 512);
        assert.strictEqual(body.choices[0].message.content, 'Raw response.');

        console.log('raw-nim.test.js: all assertions passed');
    } catch (error) {
        console.error(error);
        process.exitCode = 1;
    } finally {
        axios.post = originalPost;
        server.close();
    }
});

server.on('error', function(error) {
    axios.post = originalPost;
    throw error;
});
