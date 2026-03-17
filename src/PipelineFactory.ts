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
 */

/**
 * Shared ONNX runtime configuration.
 * Detects the execution environment and configures the appropriate backend:
 *   - Node.js → onnxruntime-node + filesystem cache
 *   - Browser → built-in WebGPU/WASM (handled by transformers.js)
 *
 * All downstream modules (model.js, Embedder.js) import from here
 * so environment setup runs exactly once.
 */
const isNode = hasPath(process, 'versions', 'node')
if (isNode) {
    console.debug('Running in Node.js environment, configuring ONNX Runtime for Node')
} else {
    console.debug('Running in browser environment, using built-in ONNX Runtime with WebGPU/WASM backends')
}
if (isNode) {
    // Dynamic import prevents bundlers from resolving onnxruntime-node in browser builds.
    // Using a variable defeats static analysis for bundlers like esbuild.
    const ort = await import('onnxruntime-node')
    env.backends.onnx.runtime = ort.default ?? ort
    env.cacheDir = './.cache'
}

/**
 * Returns the best available compute device for the current environment.
 * Node.js always uses CPU. Browsers prefer WebGPU with WASM fallback.
 * @returns {Promise<string>} "webgpu" or "wasm" for browsers, "cpu" for Node.js
 */
export async function getDevice(): Promise<'webgpu' | 'wasm' | 'cpu'> {
    if (isNode) {
        return 'cpu'
    }
    try {
        const adapter = await (navigator as any)?.gpu?.requestAdapter()
        return adapter ? 'webgpu' : 'wasm'
    } catch {
        return 'wasm'
    }
}

export async function createPipeline<TTask extends PipelineType>(task: TTask, model: string, options: PretrainedModelOptions = {}) {
    const device = await getDevice()
    console.debug(`Creating pipeline for task ${task} and model ${model} on device ${device}`)
    return await pipeline(task, model, {
        device,
        progress_callback: pipelineProgressConsoleReporter,
        ...options,
    })
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
            const instance = await createPipeline<TTask>(this.#task, this.#modelId, this.#pipelineOptions)
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
