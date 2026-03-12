import { Message } from '@huggingface/transformers'
import { isNum } from 'jty'

function simpleProgressBar(percent: number): string {
    if (!isNum(percent)) {
        return `${percent}%`
    }
    percent = clamp(percent, 0, 100)
    const filled = '█'.repeat(Math.round(percent / 10))
    const empty = '░'.repeat(10 - filled.length)
    return `${filled}${empty} ${percent.toFixed(2)}%`
}

interface PipelineProgressInitiateEvent {
    status: 'initiate'
    name: string
    file: string
}

interface PipelineProgressDownloadEvent {
    status: 'download'
    name: string
    file: string
}

interface PipelineProgressProgressEvent {
    status: 'progress'
    /** The name of the file being processed. */
    name: string
    /** The specific file path or URL. */
    file: string
    /** The completion percentage */
    progress: number
    /** The total byte size expected. It may change during progress. */
    total: number
}

interface PipelineProgressReadyEvent {
    status: 'ready'
    /** The overall pipeline task type (e.g. 'text-generation') */
    task: string
    /** The model name being loaded */
    model: string
}

interface PipelineProgressDoneEvent {
    status: 'done'
    name: string
    file: string
}

/**
 * An object detailing the current progress of a pipeline initialization step.
 * A single name like 'Xenova/all-MiniLM-L6-v2' may initiate downloading and loading
 * multiple files like 'tokenizer_config.json', 'config.json', 'onnx/model_quantized.onnx', etc.
 */
type PipelineProgressEvent =
    | PipelineProgressInitiateEvent
    | PipelineProgressDownloadEvent
    | PipelineProgressProgressEvent
    | PipelineProgressReadyEvent
    | PipelineProgressDoneEvent

/**
 * Callback function designed to handle progress events emitted by Transformers.js during model loading.
 * Logs informative console messages for file downloads and initialization phases.
 */
export function pipelineProgressConsoleReporter(progressEvent: PipelineProgressEvent) {
    switch (progressEvent.status) {
        case 'initiate':
            /**
             * The library has realized it needs a specific file and has started the process of fetching it.
             * This happens before any data transfer begins.
             * Defined: name, file
             */
            console.debug(`File Initiate:\nName: ${progressEvent.name}, File: ${progressEvent.file}`)
            break
        case 'download':
            /**
             * Fired at the start of the download.
             * Defined: name, file
             */
            console.debug(`File Download:\nName: ${progressEvent.name}, File: ${progressEvent.file}`)
            break
        case 'progress':
            /**
             * The file is being downloaded OR being transferred into memory.
             * Fired with progress=100 if the file is already downloaded
             * Defined: name, file, progress, total
             */
            console.debug(
                `File Progress:\nName: ${progressEvent.name}, File: ${progressEvent.file}, Progress: ${simpleProgressBar(progressEvent.progress)} Total: ${progressEvent.total}`,
            )
            break
        case 'ready':
            /**
             * A special final event. It signifies that all necessary files have been
             * loaded, parsed, and the model is fully initialized and ready for inference.
             * Defined: task, model
             */
            console.debug(`Task Ready:\nTask: ${progressEvent.task}, Model ${progressEvent.model}`)
            break
        case 'done':
            /**
             * The bytes for this individual file have been fully loaded into memory.
             * Defined: name, file
             */
            console.debug(`File Done:\nName: ${progressEvent.name}, File: ${progressEvent.file}`)
            break
        default:
            console.warn(`Progress event with unknown status: ${JSON.stringify(progressEvent)}`)
    }
}

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
