import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { parsePythonToolCallObj, stripToolCallTokens, toolCallObjArgumentsToStr } from './liquid-tools-transpiler'

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

describe(toolCallObjArgumentsToStr.name, () => {
    test('converts string arguments to quoted format', () => {
        assert.strictEqual(toolCallObjArgumentsToStr('{"arg1":"hello"}'), 'arg1="hello"')
        const valueWithQuotes = 'he"llo'
        assert.strictEqual(toolCallObjArgumentsToStr(JSON.stringify({ arg1: valueWithQuotes })), 'arg1="he\\"llo"')

    })

    test('converts number arguments', () => {
        assert.strictEqual(toolCallObjArgumentsToStr('{"arg1":42}'), 'arg1=42')
    })

    test('converts booleans to Python format', () => {
        assert.strictEqual(toolCallObjArgumentsToStr('{"arg1":true}'), 'arg1=True')
        assert.strictEqual(toolCallObjArgumentsToStr('{"arg1":false}'), 'arg1=False')
    })

    test('converts null to Python None', () => {
        assert.strictEqual(toolCallObjArgumentsToStr('{"arg1":null}'), 'arg1=None')
    })

    test('converts multiple arguments', () => {
        assert.strictEqual(
            toolCallObjArgumentsToStr('{"name":"alice","age":30,"active":true}'),
            'name="alice", age=30, active=True',
        )
    })

    test('returns empty string for undefined input', () => {
        assert.strictEqual(toolCallObjArgumentsToStr(undefined), '')
        assert.strictEqual(toolCallObjArgumentsToStr(''), '')
    })
})