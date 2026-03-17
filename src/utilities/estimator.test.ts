import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { Estimator, _test } from './estimator'

const { getAverageSpeed } = _test

describe(_test.getAverageSpeed.name, () => {
    test('calculates average speed from data points', () => {
        const dp = (timestamp: number, progress: number) => ({ timestamp, progress })

        const data = [
            dp(1000, 0),
            dp(2000, 0.25),
            dp(3000, 0.5),
        ]

        assert.strictEqual(getAverageSpeed(data), 0.25 / 1000)
    })

    test('throws a TypeError if the input is not an array', () => {
        assert.throws(() => getAverageSpeed('not an array' as any), {
            name: 'TypeError',
            message: 'Expected an array. Got not an array (string)',
        })
        assert.throws(() => getAverageSpeed(null as any), TypeError)
        assert.throws(() => getAverageSpeed(undefined as any), TypeError)
        assert.throws(() => getAverageSpeed(123 as any), TypeError)
        assert.throws(() => getAverageSpeed({} as any), TypeError)
    })

    test('throws if the input has less than 2 elements', () => {
        assert.throws(() => getAverageSpeed([]), {
            name: 'Error',
            message: 'Not enough data points to calculate speed.',
        })

        assert.throws(() => getAverageSpeed([{ progress: 0, timestamp: 10 }]), {
            name: 'Error',
            message: 'Not enough data points to calculate speed.',
        })
    })
})

describe(Estimator.name, () => {
    test('calculates remaining time correctly', () => {
        const estimator = new Estimator()
        estimator.report(0, 1000)
        estimator.report(0.25, 2000)
        estimator.report(0.5, 3000)

        assert.strictEqual(estimator.remaining, 2000)
    })

    test('throws if invalid data is being pushed', () => {
        const estimator = new Estimator()

        assert.throws(() => estimator.report('1' as any), {
            name: 'TypeError',
            message: 'Expected a finite number for progress. Got 1 (string)',
        })

        assert.throws(() => estimator.report(100), {
            name: 'RangeError',
            message: 'Progress (100) must be between 0 and 1 (inclusive)',
        })
    })
})
