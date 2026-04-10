import { isFin, isNum } from 'jty'

// -- Intl.DurationFormat is not yet in TypeScript's lib definitions --

interface DurationParts {
    hours: number
    minutes: number
    seconds: number
}

interface DurationFormatOptions {
    style?: 'long' | 'short' | 'narrow' | 'digital'
}

// -- Constants --

const DEFAULT_DURATION_STYLE: DurationFormatOptions = { style: 'long' }

// -- Public API --

export function bytesToHumanReadable(bytes: number): string {
    if (!isNum(bytes)) {
        throw new TypeError(`Expected a non-negative number. Got ${bytes} (${typeof bytes})`)
    }

    const sign = bytes < 0 ? '-' : ''
    const absBytes = Math.abs(bytes)

    if (absBytes < 1024) {
        return `${sign}${absBytes} B`
    }

    const units = ['KB', 'MB', 'GB']
    let unitIndex = -1
    let value = absBytes

    while (value >= 1024 && unitIndex < units.length - 1) {
        value /= 1024
        unitIndex++
    }

    return `${sign}${value.toFixed(2)} ${units[unitIndex]}`
}

/**
 * Formats a number into a human-readable, localized string.
 *
 * @param value - The number to format.
 * @param locale - The locale to use for formatting. Defaults to the runtime's locale.
 * @param options - Options for {@link Intl.NumberFormat}.
 * @returns The formatted number string, or an empty string for non-finite input.
 *
 * @example
 * ```ts
 * numL10n(1234.5, 'en-US') // '1,234.5'
 * numL10n(1234.5, 'de-DE') // '1.234,5'
 * numL10n(0.75, 'en-US', { style: 'percent' }) // '75%'
 * ```
 */
export function numL10n(value: number, locale?: string, options?: Intl.NumberFormatOptions): string {
    if (!isFin(value)) {
        return ''
    }

    return new Intl.NumberFormat(locale, options).format(value)
}

/**
 * Formats a duration in milliseconds into a human-readable, localized string
 * using {@link Intl.DurationFormat}.
 *
 * @param milliseconds - The duration in milliseconds (must be non-negative).
 * @param locale - The locale to use for formatting. Defaults to the runtime's locale.
 * @param options - Options for {@link Intl.DurationFormat}. Defaults to `{ style: 'long' }`.
 * @returns The formatted duration string, or an empty string for invalid input.
 *
 * @example
 * ```ts
 * durL10n(3_661_000, 'en-US') // '1 hour, 1 minute, 1 second'
 * durL10n(90_000, 'en-US')    // '1 minute, 30 seconds'
 * durL10n(500, 'en-US')       // '1 second'
 * ```
 */
export function durL10n(
    milliseconds: number,
    locale?: string,
    options: DurationFormatOptions = DEFAULT_DURATION_STYLE,
): string {
    if (!isFin(milliseconds) || milliseconds < 0) {
        return ''
    }

    const duration = millisecondsToDurationParts(milliseconds)

    try {
        const formatter = new Intl.DurationFormat(locale, options)
        return formatter.format(duration)
    } catch {
        return formatDurationFallback(duration)
    }
}

// -- Internal helpers --

/**
 * Converts a millisecond value to hours, minutes, and seconds.
 *
 * @param ms - A non-negative number of milliseconds.
 * @returns An object with `hours`, `minutes`, and `seconds` fields.
 *
 * @example
 * ```ts
 * millisecondsToDurationParts(3_661_000) // { hours: 1, minutes: 1, seconds: 1 }
 * ```
 */
function millisecondsToDurationParts(ms: number): DurationParts {
    const totalSeconds = Math.round(ms / 1000)

    return {
        hours: Math.floor(totalSeconds / 3600),
        minutes: Math.floor((totalSeconds % 3600) / 60),
        seconds: totalSeconds % 60,
    }
}

/**
 * Simple fallback formatter for environments without `Intl.DurationFormat`.
 *
 * @example
 * ```ts
 * formatDurationFallback({ hours: 1, minutes: 2, seconds: 3 }) // '1h 2m 3s'
 * ```
 */
function formatDurationFallback(duration: DurationParts): string {
    return `${duration.hours}h ${duration.minutes}m ${duration.seconds}s`
}

// -- Test-only exports --

export const _test = { millisecondsToDurationParts, formatDurationFallback }
