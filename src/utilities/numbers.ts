import { isNum } from "jty"

/**
 * Clamps a number between a minimum and maximum value.
 *
 * @param val The value to clamp.
 * @param min The minimum allowed value.
 * @param max The maximum allowed value.
 * @returns The clamped value.
 */
export function clamp(val: number, min: number, max: number): number {
    if (!isNum(val)) {
        throw new TypeError('val must be a number')
    }
    if (!isNum(min)) {
        throw new TypeError('min must be a number')
    }
    if (!isNum(max)) {
        throw new TypeError('max must be a number')
    }
    if (min > max) {
        throw new TypeError('min must be less than or equal to max')
    }
    return Math.min(Math.max(val, min), max)
}

/**
 * Converts seconds to milliseconds.
 *
 * @param seconds The time in seconds.
 * @returns The time in milliseconds.
 */
export function sec2ms(seconds: number): number {
    if (!isNum(seconds)) {
        throw new TypeError('seconds must be a number')
    }
    return seconds * 1000
}
