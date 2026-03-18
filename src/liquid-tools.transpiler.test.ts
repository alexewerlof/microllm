import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import {
    convertSupportedMessagesToLiquidMessages,
    parsePythonToolCallObj,
    stripToolCallTokens,
    tryParseToolCalls,
} from './liquid-tools-transpiler.js'

describe(stripToolCallTokens.name, () => {
    test('removes the special tokens', () => {
        assert.strictEqual(stripToolCallTokens('Hello <|tool_call_start|>world<|tool_call_end|>!'), 'Hello world!')
    })
})

describe(parsePythonToolCallObj.name, () => {
    test('parses a valid tool call', () => {
        const result = parsePythonToolCallObj('[my_function(arg1="value1", arg2=\'value2\')]')
        assert.strictEqual(result.length, 1)
        assert.strictEqual(result[0].type, 'function')
        assert.strictEqual(result[0].function.name, 'my_function')
        const args = JSON.parse(result[0].function.arguments)
        assert.strictEqual(args.arg1, 'value1')
        assert.strictEqual(args.arg2, 'value2')
    })

    test('parses a tool call with no arguments', () => {
        const result = parsePythonToolCallObj('[my_function()]')
        assert.strictEqual(result.length, 1)
        assert.strictEqual(result[0].type, 'function')
        assert.strictEqual(result[0].function.name, 'my_function')
        const args = JSON.parse(result[0].function.arguments)
        assert.deepStrictEqual(args, {})
    })

    test('throws an error for invalid format', () => {
        assert.throws(() => parsePythonToolCallObj('not a tool call'), SyntaxError)
        assert.throws(() => parsePythonToolCallObj('[invalid_format]'), SyntaxError)
    })

    test('parses string arguments', () => {
        const result = parsePythonToolCallObj('[fn(name="alice")]')
        const args = JSON.parse(result[0].function.arguments)
        assert.strictEqual(args.name, 'alice')
    })

    test('parses number arguments', () => {
        const result = parsePythonToolCallObj('[fn(count=42)]')
        const args = JSON.parse(result[0].function.arguments)
        assert.strictEqual(args.count, 42)
    })

    test('parses Python True and False', () => {
        const result = parsePythonToolCallObj('[fn(a=True, b=False)]')
        const args = JSON.parse(result[0].function.arguments)
        assert.strictEqual(args.a, true)
        assert.strictEqual(args.b, false)
    })

    test('parses Python None', () => {
        const result = parsePythonToolCallObj('[fn(x=None)]')
        const args = JSON.parse(result[0].function.arguments)
        assert.strictEqual(args.x, null)
    })

    test('parses mixed argument types', () => {
        const result = parsePythonToolCallObj('[fn(name="alice", age=30, active=True, deleted=False, note=None)]')
        const args = JSON.parse(result[0].function.arguments)
        assert.strictEqual(args.name, 'alice')
        assert.strictEqual(args.age, 30)
        assert.strictEqual(args.active, true)
        assert.strictEqual(args.deleted, false)
        assert.strictEqual(args.note, null)
    })
})

describe(tryParseToolCalls.name, () => {
    test('parses a tool call wrapped in special tokens', () => {
        const result = tryParseToolCalls('<|tool_call_start|>[get_time()]<|tool_call_end|>')
        assert.notStrictEqual(result, null)
        assert.strictEqual(result!.role, 'assistant')
        assert.strictEqual(result!.tool_calls[0].function.name, 'get_time')
    })

    test('parses a tool call with surrounding text', () => {
        const result = tryParseToolCalls(
            '<|tool_call_start|>[search(query="hello")]<|tool_call_end|>\nSearching for hello.',
        )
        assert.notStrictEqual(result, null)
        assert.strictEqual(result!.tool_calls[0].function.name, 'search')
    })

    test('returns null for regular text', () => {
        assert.strictEqual(tryParseToolCalls('Hello world'), null)
    })

    test('returns null for text without special tokens', () => {
        assert.strictEqual(tryParseToolCalls('[get_time()]'), null)
    })

    test('throws when start token is present but end token is missing', () => {
        assert.throws(() => tryParseToolCalls('<|tool_call_start|>[get_time()]'), SyntaxError)
    })

    test('throws when brackets are missing inside tokens', () => {
        assert.throws(() => tryParseToolCalls('<|tool_call_start|>get_time()<|tool_call_end|>'), SyntaxError)
    })

    test('parses a tool call with arguments', () => {
        const result = tryParseToolCalls('<|tool_call_start|>[search(query="hello")]<|tool_call_end|>')
        assert.notStrictEqual(result, null)
        const args = JSON.parse(result!.tool_calls[0].function.arguments)
        assert.strictEqual(args.query, 'hello')
    })

    test('throws when text is not a string', () => {
        assert.throws(
            // @ts-expect-error testing invalid input
            () => tryParseToolCalls(123),
            TypeError,
        )
    })
})

describe(convertSupportedMessagesToLiquidMessages.name, () => {
    test('converts tool call messages into python-style assistant content', () => {
        const result = convertSupportedMessagesToLiquidMessages([
            {
                role: 'assistant',
                tool_calls: [{ id: 'call_123', type: 'function', function: { name: 'get_time', arguments: '{}' } }],
            },
        ])

        assert.deepStrictEqual(result, [
            { role: 'assistant', content: '<|tool_call_start|>[get_time()]<|tool_call_end|>' },
        ])
    })
})
