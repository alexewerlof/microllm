import { isNum } from 'jty'
import { clamp } from './numbers.js'
import { bytesToHumanReadable } from './format.js'
import { ProgressInfo } from '@huggingface/transformers'

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
 * Callback function designed to handle progress events emitted by Transformers.js during model loading.
 * Logs informative console messages for file downloads and initialization phases.
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
export function pipelineProgressConsoleReporter(progressEvent: ProgressInfo) {
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
