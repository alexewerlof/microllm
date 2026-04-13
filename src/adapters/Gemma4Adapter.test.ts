import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import { _test, Gemma4Adapter } from './Gemma4Adapter.js'
import { SupportedMessage } from '../Message/types.js'

const {
    stripGemma4Thinking,
    parseGemma4Arguments,
    tryParseGemma4ToolCallsMessage,
    convertSupportedMessagesToGemma4Messages,
    GEMMA4_FALLBACK_CHAT_TEMPLATE,
} = _test

describe(_test.stripGemma4Thinking.name, () => {
    test('removes thought channel blocks', () => {
        const input = 'Hi<|channel>thought\nreasoning<channel|> there'
        assert.strictEqual(stripGemma4Thinking(input), 'Hi there')
    })
})

describe(_test.parseGemma4Arguments.name, () => {
    test('parses quoted and numeric values', () => {
        const args = parseGemma4Arguments('location:<|"|>Seoul<|"|>,count:2')
        assert.deepStrictEqual(args, { location: 'Seoul', count: 2 })
    })
})

describe(_test.tryParseGemma4ToolCallsMessage.name, () => {
    test('parses a tool call from Gemma 4 tokens', () => {
        const message = tryParseGemma4ToolCallsMessage(
            '<|tool_call>call:get_current_weather{location:<|"|>Tokyo, JP<|"|>}<tool_call|>',
        )

        assert.strictEqual(message.role, 'assistant')
        assert.strictEqual(message.tool_calls.length, 1)
        assert.strictEqual(message.tool_calls[0].function.name, 'get_current_weather')
        assert.deepStrictEqual(JSON.parse(message.tool_calls[0].function.arguments), { location: 'Tokyo, JP' })
    })
})

describe(_test.convertSupportedMessagesToGemma4Messages.name, () => {
    test('converts canonical tool result messages into assistant tool_responses', () => {
        const messages: SupportedMessage[] = [
            {
                role: 'assistant',
                tool_calls: [
                    {
                        id: 'call_1',
                        type: 'function',
                        function: {
                            name: 'get_time',
                            arguments: '{}',
                        },
                    },
                ],
            },
            {
                role: 'tool',
                tool_call_id: 'call_1',
                content: '"noon"',
            },
        ]

        const converted = convertSupportedMessagesToGemma4Messages(messages)
        const toolResponseMessage = converted[1] as any

        assert.strictEqual(toolResponseMessage.role, 'assistant')
        assert.deepStrictEqual(toolResponseMessage.tool_responses, [{ name: 'get_time', response: 'noon' }])
    })
})

describe(Gemma4Adapter.name, () => {
    test('provides a chat_template fallback for transformers.js', () => {
        const adapter = new Gemma4Adapter()

        const prepared = adapter.onBeforeChatTemplate({
            messages: [{ role: 'user', content: 'Hello' }],
        })

        assert.strictEqual(prepared.templateOptions.chat_template, GEMMA4_FALLBACK_CHAT_TEMPLATE)
    })

    test('returns assistant message when output has no tool call tokens', () => {
        const adapter = new Gemma4Adapter()

        const message = adapter.onAfterDecode({
            rawAssistantContent: 'Regular answer',
            cleanAssistantContent: 'Regular answer',
        })

        assert.deepStrictEqual(message, {
            role: 'assistant',
            content: 'Regular answer',
        })
    })
})
