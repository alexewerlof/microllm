import { isArr, isNum, isArrLen } from 'jty'

interface DataPoint {
    progress: number
    timestamp: number
}

/**
 * Calculates the average speed of progress over time from a series of data points.
 *
 * @example
 * ```ts
 * const points = [
 *   { progress: 0, timestamp: 1000 },
 *   { progress: 0.5, timestamp: 2000 },
 * ]
 * getAverageSpeed(points) // 0.0005 (progress per ms)
 * ```
 *
 * @param dataPoints an array of at least 2 data points with `progress` and `timestamp`
 * @returns the average speed in progress-per-millisecond
 */
function getAverageSpeed(dataPoints: DataPoint[]): number {
    if (!isArr(dataPoints)) {
        throw new TypeError(`Expected an array. Got ${dataPoints} (${typeof dataPoints})`)
    }

    if (!isArrLen(dataPoints, 2)) {
        throw new Error('Not enough data points to calculate speed.')
    }

    const speeds: number[] = []

    for (let i = 1; i < dataPoints.length; i++) {
        const curr = dataPoints[i]
        const prev = dataPoints[i - 1]

        const deltaTimestamp = curr.timestamp - prev.timestamp
        const deltaProgress = curr.progress - prev.progress

        if (deltaTimestamp > 0 && deltaProgress >= 0) {
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
 * @example
 * ```ts
 * const estimator = new Estimator()
 * estimator.report(0.25)
 * estimator.report(0.5)
 * const ms = estimator.remaining // estimated ms until progress reaches 1
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
     * const e = new Estimator()
     * e.report(0.5, 2000)
     * e.lastDataPoint // { progress: 0.5, timestamp: 2000 }
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
     * e.report(0.5)
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
     * const e = new Estimator()
     * e.report(0.25, 1000).report(0.5, 2000)
     * ```
     *
     * @param progress a number between 0 and 1 (inclusive)
     * @param timestamp the timestamp in milliseconds
     * @returns this instance for chaining
     */
    report(progress: number, timestamp: number = Date.now()): this {
        if (!isNum(progress)) {
            throw new TypeError(`Expected a finite number for progress. Got ${progress} (${typeof progress})`)
        }

        if (progress < 0 || progress > 1) {
            throw new RangeError(`Progress (${progress}) must be between 0 and 1 (inclusive)`)
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
     * e.report(0.5)
     * e.isReady   // true
     * ```
     */
    get isReady(): boolean {
        return this.#dataPoints.length >= MIN_DATA_POINTS
    }

    /**
     * Estimated milliseconds remaining until progress reaches 1.
     *
     * @example
     * ```ts
     * const e = new Estimator()
     * e.report(0.25, 2000)
     * e.report(0.5, 3000)
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

        if (progress === 1) {
            return 0
        }

        return (1 - progress) / averageSpeed
    }
}

export const _test = { getAverageSpeed }
