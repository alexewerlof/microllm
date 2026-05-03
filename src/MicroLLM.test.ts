import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import { isMicroLLM } from './MicroLLM.js'

describe(isMicroLLM.name, () => {
    test('returns true for objects exposing complete()', () => {
        assert.strictEqual(
            isMicroLLM({
                async complete() {
                    return { role: 'assistant', content: 'ok' }
                },
            }),
            true,
        )
    })

    test('returns false for non-objects', () => {
        assert.strictEqual(isMicroLLM('llm'), false)
    })

    test('returns false when complete is missing', () => {
        assert.strictEqual(isMicroLLM({}), false)
    })

    test('returns false when complete is not a function', () => {
        assert.strictEqual(isMicroLLM({ complete: true }), false)
    })
})
