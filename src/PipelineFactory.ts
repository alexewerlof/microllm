import { hasPath, isObj, isStr } from 'jty'
import { pipeline, env, PipelineType, AllTasks, PretrainedModelOptions } from '@huggingface/transformers'
import { pipelineProgressConsoleReporter } from './utilities/download.js'

/*
 * Tips for future development:
 * Disable the loading of remote models from the Hugging Face Hub:
 * env.allowRemoteModels = false;
 *
 * Set location of .wasm files. Defaults to use a CDN.
 * env.backends.onnx.wasm.wasmPaths = '/path/to/files/';
 *
 * By default, unless you pass { local_files_only: true }, transformers.js will send a tiny,
 * lightweight HEAD request to the Hugging Face Hub to check the ETag of the file.
 * If the cache is still up to date, it instantly falls back to reading from the local cache,
 * firing the "initiated" -> "download" -> "progress" (instantly 100%) -> "done" events.
 * Specify a custom location for models (defaults to '/models/').
 * env.localModelPath = "/huggingface";
 *
 * Enable WASM Caching for offline-first scenarios
 * env.useWasmCache = true;
 *
 * Add to suppress ONNX Runtime warnings and console noise.
 * env.logLevel = 'error'
 */

if (hasPath(globalThis, 'process', 'versions', 'node')) {
    // Running in node.js
    env.cacheDir = './.cache'
} else {
    console.debug('Running in browser environment, using built-in ONNX Runtime with WebGPU/WASM backends')
}

export class PipelineFactory<TTask extends PipelineType> {
    #task: TTask
    #pipeline: Promise<AllTasks[TTask]> | null = null
    #modelId: string
    #pipelineOptions: PretrainedModelOptions

    constructor(task: TTask, modelId: string, pipelineOptions: PretrainedModelOptions = {}) {
        if (!isStr(task)) {
            throw new TypeError(`Expected string for task, but got ${task} (${typeof task})`)
        }
        this.#task = task
        if (!isStr(modelId)) {
            throw new TypeError(`Expected string for modelId, but got ${modelId} (${typeof modelId})`)
        }
        this.#modelId = modelId
        if (!isObj(pipelineOptions)) {
            throw new TypeError(
                `Expected object for pipelineOptions, but got ${pipelineOptions} (${typeof pipelineOptions})`,
            )
        }
        this.#pipelineOptions = pipelineOptions
    }

    get isLoaded() {
        return this.#pipeline !== null
    }

    async getPipeline() {
        this.#pipeline ??= this.#load()
        return this.#pipeline
    }

    async #load() {
        const logLine = `Create pipeline for task ${this.#task} and model ${this.#modelId}`
        try {
            console.time(logLine)
            console.debug(`Creating pipeline for task ${this.#task} and model ${this.#modelId}`)
            const instance = await pipeline(this.#task, this.#modelId, {
                progress_callback: pipelineProgressConsoleReporter,
                ...this.#pipelineOptions,
            })
            console.timeEnd(logLine)
            return instance
        } catch (error) {
            this.#pipeline = null
            throw new Error(`${logLine}: ${error}`, { cause: error })
        }
    }

    async unload() {
        if (this.#pipeline !== null) {
            const instance = await this.#pipeline
            this.#pipeline = null
            await instance.dispose()
        }
    }
}
