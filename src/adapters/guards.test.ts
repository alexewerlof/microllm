import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import { LiquidAdapter } from './LiquidAdapter.js'
import { isChatModelAdapter } from './guards.js'

describe(isChatModelAdapter.name, () => {
    test('returns true for adapter-shaped plain objects', () => {
        assert.strictEqual(
            isChatModelAdapter({
                onBeforeChatTemplate() {
                    return { messages: [], templateOptions: {} }
                },
                onAfterDecode() {
                    return { role: 'assistant', content: 'ok' }
                },
            }),
            true,
        )
    })

    test('returns true for class-based adapters', () => {
        assert.strictEqual(isChatModelAdapter(new LiquidAdapter()), true)
    })

    test('returns false for non-objects', () => {
        assert.strictEqual(isChatModelAdapter('adapter'), false)
    })

    test('returns false when required hooks are missing', () => {
        assert.strictEqual(
            isChatModelAdapter({
                onBeforeChatTemplate() {
                    return { messages: [], templateOptions: {} }
                },
            }),
            false,
        )
    })

    test('returns false when required hooks are not functions', () => {
        assert.strictEqual(
            isChatModelAdapter({
                onBeforeChatTemplate: true,
                onAfterDecode: 'nope',
            }),
            false,
        )
    })
})