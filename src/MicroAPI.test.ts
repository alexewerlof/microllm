import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import { MicroAPI } from './MicroAPI.js'

function createJsonResponse(body: unknown, status = 200) {
    return {
        ok: status >= 200 && status < 300,
        status,
        async json() {
            return body
        },
        async text() {
            return JSON.stringify(body)
        },
    }
}

describe(MicroAPI.name, () => {
    test('throws for invalid baseUrl', () => {
        assert.throws(() => new MicroAPI({ baseUrl: 'not-a-url', modelId: 'test-model' }), /Invalid URL/)
    })

    test('posts chat completion and maps assistant text', async () => {
        const oldFetch = globalThis.fetch
        let capturedHeaders: Headers | undefined
        let capturedBody: Record<string, unknown> | undefined

        globalThis.fetch = (async (_url, init) => {
            capturedHeaders = init?.headers as Headers
            capturedBody = JSON.parse(String(init?.body))
            return createJsonResponse({
                id: 'chatcmpl_1',
                object: 'chat.completion',
                created: 1,
                model: 'local-model',
                choices: [
                    {
                        index: 0,
                        message: { role: 'assistant', content: 'Hello from API' },
                        finish_reason: 'stop',
                    },
                ],
                usage: {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                },
            }) as Response
        }) as typeof fetch

        try {
            const api = new MicroAPI({
                baseUrl: 'http://localhost:1234/v1',
                modelId: 'local-model',
                apiKey: 'secret',
                useApiKey: true,
            })

            const result = await api.complete({
                messages: [{ role: 'user', content: 'Hello' }],
                config: { temperature: 0.7 },
            })

            assert.strictEqual(capturedHeaders?.get('Authorization'), 'Bearer secret')
            assert.strictEqual(capturedBody?.model, 'local-model')
            assert.strictEqual(capturedBody?.temperature, 0.7)
            assert.strictEqual(result.role, 'assistant')
            assert.strictEqual(result.content, 'Hello from API')
        } finally {
            globalThis.fetch = oldFetch
        }
    })

    test('maps OpenAI tool_calls into ToolCallsMessage', async () => {
        const oldFetch = globalThis.fetch

        globalThis.fetch = (async () => {
            return createJsonResponse({
                id: 'chatcmpl_1',
                object: 'chat.completion',
                created: 1,
                model: 'local-model',
                choices: [
                    {
                        index: 0,
                        message: {
                            role: 'assistant',
                            content: null,
                            tool_calls: [
                                {
                                    id: 'call_1',
                                    type: 'function',
                                    function: {
                                        name: 'get_time',
                                        arguments: '{"timezone":"UTC"}',
                                    },
                                },
                            ],
                        },
                        finish_reason: 'tool_calls',
                    },
                ],
                usage: {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                },
            }) as Response
        }) as typeof fetch

        try {
            const api = new MicroAPI({
                baseUrl: 'http://localhost:1234/v1',
                modelId: 'local-model',
            })

            const result = await api.complete({
                messages: [{ role: 'user', content: 'What time is it?' }],
            })

            assert.strictEqual(result.role, 'assistant')
            assert.ok('tool_calls' in result)
            assert.strictEqual(result.tool_calls[0].function.name, 'get_time')
        } finally {
            globalThis.fetch = oldFetch
        }
    })

    test('surfaces non-OK responses with request context', async () => {
        const oldFetch = globalThis.fetch

        globalThis.fetch = (async () => {
            return createJsonResponse({ error: { message: 'failure' } }, 500) as Response
        }) as typeof fetch

        try {
            const api = new MicroAPI({
                baseUrl: 'http://localhost:1234/v1',
                modelId: 'local-model',
            })

            await assert.rejects(() => api.complete({ messages: [{ role: 'user', content: 'Hi' }] }), {
                message: /Failed POST request to/,
            })
        } finally {
            globalThis.fetch = oldFetch
        }
    })

    test('accepts OpenAI-style wrapped completion response', async () => {
        const oldFetch = globalThis.fetch

        globalThis.fetch = (async () => {
            return createJsonResponse({
                id: 'chatcmpl_1',
                object: 'chat.completion',
                created: 1,
                model: 'local-model',
                choices: [
                    {
                        index: 0,
                        message: {
                            role: 'assistant',
                            content: 'Wrapped response',
                        },
                        finish_reason: 'stop',
                    },
                ],
                usage: {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                },
            }) as Response
        }) as typeof fetch

        try {
            const api = new MicroAPI({
                baseUrl: 'http://localhost:1234/v1',
                modelId: 'local-model',
            })

            const result = await api.complete({
                messages: [{ role: 'user', content: 'Hello' }],
            })

            assert.strictEqual(result.role, 'assistant')
            assert.strictEqual(result.content, 'Wrapped response')
        } finally {
            globalThis.fetch = oldFetch
        }
    })

    test('throws when response does not conform to ChatCompletionsResponse', async () => {
        const oldFetch = globalThis.fetch

        globalThis.fetch = (async () => {
            return createJsonResponse({ choices: [] }) as Response
        }) as typeof fetch

        try {
            const api = new MicroAPI({
                baseUrl: 'http://localhost:1234/v1',
                modelId: 'local-model',
            })

            await assert.rejects(() => api.complete({ messages: [{ role: 'user', content: 'Hi' }] }), {
                message: /Expected response to conform to ChatCompletionsResponse/,
            })
        } finally {
            globalThis.fetch = oldFetch
        }
    })
})
