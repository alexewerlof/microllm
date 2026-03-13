import { Message, PretrainedModelOptions, TextGenerationConfig, TextStreamer } from "@huggingface/transformers"
import { TransformersPipelineFactory } from "./TransformersPipelineFactory"
import { hasProp, isArr, isDef, isObj } from "jty"
import { normalizeMessageArray } from "./normalization"
import { SignalStoppingCriteria } from "./SignalStoppingCriteria"

const defaultTextGenerationConfig: Partial<TextGenerationConfig> = {
    max_new_tokens: 512,
    temperature: 0.5,
    top_p: 0.5,
}

export class MicroLLM {
    transformersPipelineFactory: TransformersPipelineFactory<"text-generation">
    
    constructor(modelId: string, pipelineOptions: PretrainedModelOptions = {}) {
        this.transformersPipelineFactory = new TransformersPipelineFactory('text-generation', modelId, pipelineOptions)
    }

    async complete(params: {
        messages: Message[],
        config?: Partial<TextGenerationConfig>,
        signal?: AbortSignal,
    }): Promise<string> {
        if (!isObj(params)) {
            throw new TypeError(`Expected object for params, but got ${params} (${typeof params})`)
        }
        const { signal, messages, config } = params

        const textGenerationConfig: Partial<TextGenerationConfig> = {
            ...defaultTextGenerationConfig,
            return_full_text: false,
            ...config,
        }
        if (isDef(signal)) {
            if (signal.aborted) {
                throw new Error('The signal is already aborted. Cannot start generation.')
            }
            Object.assign(textGenerationConfig, {
                stopping_criteria: SignalStoppingCriteria.createStoppingCriteriaList(signal)
            })
        }

        if (!Array.isArray(messages)) {
            throw new TypeError(`Expected array for messages, but got ${messages} (${typeof messages})`)
        }
        const pipelineInstance = await this.transformersPipelineFactory.getPipeline()
        const normalizedMessages = normalizeMessageArray(messages)
        const output = await pipelineInstance(normalizedMessages, textGenerationConfig)
        if (!isArr(output)) {
            throw new TypeError(`Expected array output from pipeline, but got ${JSON.stringify(output)} (${typeof output})`)
        }
        const firstCompletion = output[0]
        if (!hasProp(firstCompletion, 'generated_text') || !isArr(firstCompletion.generated_text)) {
            throw new TypeError(`Expected output objects to have "generated_text" array, but got ${JSON.stringify(firstCompletion)}`)
        }
        return firstCompletion.generated_text[firstCompletion.generated_text.length - 1].content
    }

    async *generateTokens(params: {
        messages: Message[],
        config?: Partial<TextGenerationConfig>,
        signal?: AbortSignal,
    }): AsyncGenerator<string, void, unknown> {
        if (!isObj(params)) {
            throw new TypeError(`Expected object for params, but got ${params} (${typeof params})`)
        }
        const { signal, messages, config } = params

        const textGenerationConfig: Partial<TextGenerationConfig> = {
            ...defaultTextGenerationConfig,
            ...config,
        }
        if (isDef(signal)) {
            if (signal.aborted) {
                throw new Error('The signal is already aborted. Cannot start generation.')
            }
            Object.assign(textGenerationConfig, {
                stopping_criteria: SignalStoppingCriteria.createStoppingCriteriaList(signal)
            })
        }

        if (!Array.isArray(messages)) {
            throw new TypeError(`Expected array for messages, but got ${messages} (${typeof messages})`)
        }
        const normalizedMessages = normalizeMessageArray(messages)
        console.dir(normalizedMessages, {depth: null})

        // Async queue to bridge the callback-based streamer to an async generator
        const queue: string[] = []
        let done = false
        let error: unknown = null
        let resolveWait: (() => void) | null = null

        const pipelineInstance = await this.transformersPipelineFactory.getPipeline()
        textGenerationConfig['streamer'] = new TextStreamer(pipelineInstance.tokenizer, {
            skip_prompt: true,
            callback_function(text: string) {
                queue.push(text)
                resolveWait?.()
            },
        })

        // Start generation (runs asynchronously while we yield from the queue)
        const generatePromise = pipelineInstance(normalizedMessages, textGenerationConfig)
        
        generatePromise.then(() => {
            done = true
            resolveWait?.()
        }).catch((err: unknown) => {
            error = err
            done = true
            resolveWait?.()
        })

        // Pull tokens from the queue as they arrive
        while (true) {
            if (queue.length > 0) {
                yield queue.shift()!
            } else if (done) {
                break
            } else {
                await new Promise<void>(r => { resolveWait = r })
                resolveWait = null
            }
        }

        await generatePromise
        if (error) throw error
    }
}