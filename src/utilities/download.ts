import { isNum } from 'jty'
import { clamp } from './numbers.js'
import { bytesToHumanReadable } from './format.js'

function simpleProgressBar(percent: number): string {
    if (!isNum(percent)) {
        return `${percent}%`
    }
    percent = clamp(percent, 0, 100)
    const filled = '█'.repeat(Math.round(percent / 10))
    const empty = '░'.repeat(10 - filled.length)
    return `${filled}${empty} ${percent.toFixed(2)}%`
}

/**
 * The library has realized it needs a specific file and has started the process of downloading it.
 * This happens before any data transfer begins.
 * Defined: name, file
 */
interface PipelineProgressInitiateEvent {
    status: 'initiate'
    name: string
    file: string
}

/**
 * Fired at the start of the download (even if the file is already cached).
 * Defined: name, file
 */
interface PipelineProgressDownloadEvent {
    status: 'download'
    name: string
    file: string
}

/**
 * The file is being downloaded OR being transferred into memory.
 * Fired with progress=100 if the file is already downloaded
 * Defined: name, file, progress, total
 */
interface PipelineProgressProgressEvent {
    status: 'progress'
    /** The name of the file being processed. */
    name: string
    /** The specific file path or URL. */
    file: string
    /** The completion percentage 0-100 */
    progress: number
    /** The total byte size expected. It may change during progress. */
    total: number
}

/**
 * A special final event. It signifies that all necessary files have been
 * loaded, parsed, and the model is fully initialized and ready for inference.
 * Defined: task, model
 */
interface PipelineProgressReadyEvent {
    status: 'ready'
    /** The overall pipeline task type (e.g. 'text-generation') */
    task: string
    /** The model name being loaded */
    model: string
}

/**
 * The bytes for this individual file have been fully loaded into memory.
 * Defined: name, file
 */
interface PipelineProgressDoneEvent {
    status: 'done'
    name: string
    file: string
}

/**
 * An object detailing the current progress of a pipeline initialization step.
 * A single name like 'Xenova/all-MiniLM-L6-v2' may initiate downloading and loading
 * multiple files like 'tokenizer_config.json', 'config.json', 'onnx/model_quantized.onnx', etc.
 * The order of the events are:
 * 1. initiate (for each file)
 * 2. download (for each file)
 * 3. progress (multiple times per file)
 * 4. done (for each file)
 * 5. ready (once per task)
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
            console.debug(`File ${progressEvent.status}, Name: ${progressEvent.name}, File: ${progressEvent.file}`)
            break
        case 'download':
            console.debug(`File ${progressEvent.status}, Name: ${progressEvent.name}, File: ${progressEvent.file}`)
            break
        case 'progress':
            console.debug(
                `File ${progressEvent.status}, Name: ${progressEvent.name}, File: ${progressEvent.file}, Progress: ${simpleProgressBar(progressEvent.progress)} Total: ${bytesToHumanReadable(progressEvent.total)}`,
            )
            break
        case 'done':
            console.debug(`File ${progressEvent.status}, Name: ${progressEvent.name}, File: ${progressEvent.file}`)
            break
        case 'ready':
            console.debug(`Task ${progressEvent.status}, Task: ${progressEvent.task}, Model ${progressEvent.model}`)
            break
        default:
            console.warn(`Progress event with unknown status: ${JSON.stringify(progressEvent)}`)
    }
}
