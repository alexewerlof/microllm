import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { Estimator, _test } from './Estimator.js'

const { getAverageSpeed } = _test

describe(_test.getAverageSpeed.name, () => {
    test('calculates average speed from data points', () => {
        const dp = (timestamp: number, progress: number) => ({ timestamp, progress })

        const now = Date.now()
        const data = [dp(now, 0), dp(now + 1000, 25), dp(now + 2000, 50)]

        assert.strictEqual(getAverageSpeed(data), 25 / 1000)
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
        const now = Date.now()
        const estimator = new Estimator()
        estimator.report(0, now)
        estimator.report(25, now + 1000)
        estimator.report(50, now + 2000)

        assert.strictEqual(estimator.remaining, 2000)
    })

    test('throws if invalid data is being pushed', () => {
        const estimator = new Estimator()

        assert.throws(() => estimator.report('1' as any), {
            name: 'TypeError',
            message: 'Expected a finite number for progress. Got 1 (string)',
        })

        assert.throws(() => estimator.report(101), {
            name: 'RangeError',
            message: 'Progress (101) must be between 0 and 100 (inclusive)',
        })
    })
})
