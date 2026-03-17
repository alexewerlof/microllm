import { isArr, isNum, isArrLen, inRange } from 'jty'

interface DataPoint {
    progress: number
    timestamp: number
}

/**
 * Calculates the average speed of progress over time from a series of data points.
 *
 * @example
 * ```ts
 * const now = Date.now()
 * const points = [
 *   { progress: 0, timestamp: now },
 *   { progress: 50, timestamp: now + 1000 },
 * ]
 * getAverageSpeed(points) // 0.05 (percentage points per ms)
 * ```
 *
 * Only consecutive pairs (after sorting by timestamp) where
 * `deltaTimestamp > 0` and `deltaProgress >= 0` are included in the
 * calculation. Pairs that go backwards in progress are silently skipped.
 * If no valid pairs remain, an error is thrown.
 *
 * @param dataPoints an array of at least 2 data points with `progress` (0–100) and `timestamp`
 * @returns the average speed in percentage-points-per-millisecond
 */
function getAverageSpeed(dataPoints: DataPoint[]): number {
    if (!isArr(dataPoints)) {
        throw new TypeError(`Expected an array. Got ${dataPoints} (${typeof dataPoints})`)
    }

    if (!isArrLen(dataPoints, 2)) {
        throw new Error('Not enough data points to calculate speed.')
    }

    const sorted = dataPoints.toSorted((a, b) => a.timestamp - b.timestamp)

    const speeds: number[] = []

    for (let i = 1; i < sorted.length; i++) {
        const curr = sorted[i]
        const prev = sorted[i - 1]

        const deltaTimestamp = curr.timestamp - prev.timestamp
        if (deltaTimestamp === 0) {
            continue
        }

        const deltaProgress = curr.progress - prev.progress

        if (deltaProgress >= 0) {
            speeds.push(deltaProgress / deltaTimestamp)
        }
    }

    if (speeds.length === 0) {
        throw new Error('Not enough meaningful speed data.')
    }

    const speedSum = speeds.reduce((acc, speed) => acc + speed, 0)

    return speedSum / speeds.length
}

const MIN_DATA_POINTS = 2

/**
 * Tracks progress over time and estimates remaining duration.
 *
 * Tracks progress over time by collecting `{ progress, timestamp }` data points
 * and using the average speed to estimate how many milliseconds remain until
 * progress reaches 100.
 *
 * Progress values are percentages between 0 and 100 (inclusive), matching the
 * format emitted by `@huggingface/transformers.js`. Timestamps are Unix epoch
 * milliseconds and default to `Date.now()` when omitted.
 *
 * The API is fluent — both {@link Estimator.report | report} and
 * {@link Estimator.reset | reset} return `this` so calls can be chained.
 *
 * @example Basic usage
 * ```ts
 * const now = Date.now()
 * const estimator = new Estimator()
 * estimator.report(0, now)
 * estimator.report(25, now + 1000)
 * estimator.report(50, now + 2000)
 * estimator.remaining // 2000 (ms)
 * ```
 *
 * @example Chained reporting
 * ```ts
 * const now = Date.now()
 * const estimator = new Estimator()
 * estimator
 *   .report(0, now)
 *   .report(50, now + 1000)
 * estimator.remaining // 1000
 * ```
 *
 * @example Checking readiness before accessing the estimate
 * ```ts
 * const now = Date.now()
 * const estimator = new Estimator()
 * estimator.isReady    // false — not enough data points yet
 * estimator.report(0, now)
 * estimator.isReady    // false — still only one data point
 * estimator.report(50, now + 1000)
 * estimator.isReady    // true
 * estimator.remaining  // 1000
 * ```
 *
 * @example Resetting to start a new measurement
 * ```ts
 * const now = Date.now()
 * const estimator = new Estimator()
 * estimator.report(0, now).report(50, now + 1000)
 * estimator.reset(now + 5000)  // clears history, records progress 0 at the new timestamp
 * estimator.isReady             // false
 * ```
 */
export class Estimator {
    #dataPoints: DataPoint[]

    constructor() {
        this.#dataPoints = []
    }

    /**
     * Returns the most recently recorded data point.
     *
     * @example
     * ```ts
     * const now = Date.now()
     * const e = new Estimator()
     * e.report(50, now)
     * e.lastDataPoint // { progress: 50, timestamp: now }
     * ```
     */
    get lastDataPoint(): DataPoint {
        return this.#dataPoints[this.#dataPoints.length - 1]
    }

    /**
     * Clears all recorded data and starts fresh from progress 0.
     *
     * @example
     * ```ts
     * const e = new Estimator()
     * e.report(50)
     * e.reset() // back to progress 0
     * ```
     *
     * @param timestamp the starting timestamp in milliseconds
     * @returns this instance for chaining
     */
    reset(timestamp: number = Date.now()): this {
        this.#dataPoints = []
        this.report(0, timestamp)

        return this
    }

    /**
     * Records a progress value at a given timestamp.
     *
     * @example
     * ```ts
     * const now = Date.now()
     * const e = new Estimator()
     * e.report(25, now).report(50, now + 1000)
     * ```
     *
     * Data points do not need to be reported in strictly increasing order.
     * However, only consecutive pairs where both time and progress move forward
     * contribute to the speed estimate — pairs that go backwards are ignored.
     *
     * When the same progress value is reported consecutively, the previous data
     * point is replaced so that only the latest timestamp for that progress
     * level is kept.
     *
     * @param progress a percentage between 0 and 100 (inclusive)
     * @param timestamp the Unix epoch timestamp in milliseconds (defaults to `Date.now()`)
     * @returns this instance for chaining
     */
    report(progress: number, timestamp: number = Date.now()): this {
        if (!isNum(progress)) {
            throw new TypeError(`Expected a finite number for progress. Got ${progress} (${typeof progress})`)
        }

        if (!inRange(progress, 0, 100)) {
            throw new RangeError(`Progress (${progress}) must be between 0 and 100 (inclusive)`)
        }

        if (!isNum(timestamp)) {
            throw new TypeError(`Expected a finite number for timestamp. Got ${timestamp} (${typeof timestamp})`)
        }

        if (timestamp < 0) {
            throw new RangeError(`timestamp (${timestamp}) cannot be negative`)
        }

        if (this.lastDataPoint && this.lastDataPoint.progress === progress) {
            this.#dataPoints.pop()
        }

        this.#dataPoints.push({ progress, timestamp })

        return this
    }

    /**
     * Whether enough data has been recorded to produce an estimate.
     *
     * @example
     * ```ts
     * const e = new Estimator()
     * e.isReady   // false  (only the initial 0-progress point)
     * e.report(50)
     * e.isReady   // true
     * ```
     */
    get isReady(): boolean {
        return this.#dataPoints.length >= MIN_DATA_POINTS
    }

    /**
     * Estimated milliseconds remaining until progress reaches 100.
     *
     * @example
     * ```ts
     * const now = Date.now()
     * const e = new Estimator()
     * e.report(0, now)
     * e.report(25, now + 1000)
     * e.report(50, now + 2000)
     * e.remaining // 2000
     * ```
     */
    get remaining(): number {
        if (!this.isReady) {
            throw new Error('Not enough data points to calculate remaining time.')
        }

        const averageSpeed = getAverageSpeed(this.#dataPoints)

        if (averageSpeed <= 0) {
            throw new Error('Current data does not indicate any progress.')
        }

        const { progress } = this.lastDataPoint

        if (progress === 100) {
            return 0
        }

        return (100 - progress) / averageSpeed
    }
}

export const _test = { getAverageSpeed }
