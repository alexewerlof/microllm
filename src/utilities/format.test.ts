import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { _test, bytesToHumanReadable, numL10n, durL10n } from './format.js'

const { millisecondsToDurationParts, formatDurationFallback } = _test

function tableTest<T extends (...args: any[]) => any>(fn: T, cases: { input: Parameters<T>; output: ReturnType<T> }[]) {
    for (const { input, output } of cases) {
        assert.strictEqual(fn(...input), output)
    }
}

describe(bytesToHumanReadable.name, () => {
    test('converts bytes to human-readable format', () => {
        tableTest(bytesToHumanReadable, [
            { input: [0], output: '0 B' },
            { input: [500], output: '500 B' },
            { input: [1024], output: '1.00 KB' },
            { input: [1536], output: '1.50 KB' },
            { input: [1048576], output: '1.00 MB' },
            { input: [1073741824], output: '1.00 GB' },
        ])
    })

    test('handles negative byte values', () => {
        tableTest(bytesToHumanReadable, [
            { input: [-500], output: '-500 B' },
            { input: [-1024], output: '-1.00 KB' },
        ])
    })

    test('throws a TypeError for invalid inputs', () => {
        assert.throws(() => bytesToHumanReadable('not a number' as any), {
            name: 'TypeError',
            message: 'Expected a non-negative number. Got not a number (string)',
        })
        assert.throws(() => bytesToHumanReadable(null as any), TypeError)
        assert.throws(() => bytesToHumanReadable(undefined as any), TypeError)
        assert.throws(() => bytesToHumanReadable({} as any), TypeError)
    })
})

// -- Leaf: millisecondsToDurationParts --

describe(_test.millisecondsToDurationParts.name, () => {
    test('converts exact hour/minute/second boundaries', () => {
        assert.deepStrictEqual(millisecondsToDurationParts(3_600_000), { hours: 1, minutes: 0, seconds: 0 })
        assert.deepStrictEqual(millisecondsToDurationParts(60_000), { hours: 0, minutes: 1, seconds: 0 })
        assert.deepStrictEqual(millisecondsToDurationParts(1_000), { hours: 0, minutes: 0, seconds: 1 })
    })

    test('converts a mixed duration', () => {
        assert.deepStrictEqual(millisecondsToDurationParts(3_661_000), { hours: 1, minutes: 1, seconds: 1 })
    })

    test('returns all zeros for 0 ms', () => {
        assert.deepStrictEqual(millisecondsToDurationParts(0), { hours: 0, minutes: 0, seconds: 0 })
    })

    test('rounds sub-second values to the nearest second', () => {
        assert.deepStrictEqual(millisecondsToDurationParts(500), { hours: 0, minutes: 0, seconds: 1 })
        assert.deepStrictEqual(millisecondsToDurationParts(499), { hours: 0, minutes: 0, seconds: 0 })
    })
})

// -- Leaf: formatDurationFallback --

describe(_test.formatDurationFallback.name, () => {
    test('formats a full duration', () => {
        assert.strictEqual(formatDurationFallback({ hours: 1, minutes: 2, seconds: 3 }), '1h 2m 3s')
    })

    test('formats zeros', () => {
        assert.strictEqual(formatDurationFallback({ hours: 0, minutes: 0, seconds: 0 }), '0h 0m 0s')
    })
})

// -- numL10n --

describe(numL10n.name, () => {
    test('formats a number with a locale', () => {
        assert.strictEqual(numL10n(1234.5, 'en-US'), '1,234.5')
    })

    test('formats with Intl options', () => {
        assert.strictEqual(numL10n(0.75, 'en-US', { style: 'percent' }), '75%')
    })

    test('formats negative numbers', () => {
        assert.strictEqual(numL10n(-42, 'en-US'), '-42')
    })

    test('formats zero', () => {
        assert.strictEqual(numL10n(0, 'en-US'), '0')
    })

    test('returns empty string for NaN', () => {
        assert.strictEqual(numL10n(NaN), '')
    })

    test('returns empty string for Infinity', () => {
        assert.strictEqual(numL10n(Infinity), '')
        assert.strictEqual(numL10n(-Infinity), '')
    })

    test('returns empty string for non-number input', () => {
        assert.strictEqual(numL10n('hello' as any), '')
        assert.strictEqual(numL10n(undefined as any), '')
        assert.strictEqual(numL10n(null as any), '')
    })
})

// -- durL10n (depends on millisecondsToDurationParts + Intl.DurationFormat) --

describe(durL10n.name, () => {
    test('formats a duration in long style by default', () => {
        const result = durL10n(3_661_000, 'en-US')
        assert.strictEqual(result, '1 hour, 1 minute, 1 second')
    })

    test('formats seconds only', () => {
        assert.strictEqual(durL10n(5_000, 'en-US'), '5 seconds')
    })

    test('returns empty string for zero milliseconds', () => {
        assert.strictEqual(durL10n(0, 'en-US'), '')
    })

    test('supports short style option', () => {
        const result = durL10n(3_661_000, 'en-US', { style: 'short' })
        assert.strictEqual(result, '1 hr, 1 min, 1 sec')
    })

    test('returns empty string for negative values', () => {
        assert.strictEqual(durL10n(-1), '')
    })

    test('returns empty string for NaN', () => {
        assert.strictEqual(durL10n(NaN), '')
    })

    test('returns empty string for non-number input', () => {
        assert.strictEqual(durL10n('hello' as any), '')
        assert.strictEqual(durL10n(undefined as any), '')
    })
})
