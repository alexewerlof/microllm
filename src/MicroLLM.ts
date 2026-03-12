import { Message, PretrainedModelOptions, TextGenerationConfig, TextStreamer } from "@huggingface/transformers"
import { TransformersPipelineFactory } from "./TransformersPipelineFactory"
import { isObj } from "jty"
import { normalizeMessageArray } from "./normalization"

const defaultTextGenerationConfig: Partial<TextGenerationConfig> = {
    max_new_tokens: 128,
    temperature: 0.7,
    top_p: 0.5,
}

export class MicroLLM {
    transformersPipelineFactory: TransformersPipelineFactory<"text-generation">
    
    constructor(modelId: string, pipelineOptions: PretrainedModelOptions = {}) {
        this.transformersPipelineFactory = new TransformersPipelineFactory('text-generation', modelId, pipelineOptions)
    }

    async *chatCompletion(params: {
        messages: Message[],
        config?: Partial<TextGenerationConfig>,
    }): AsyncGenerator<string, void, unknown> {
        if (!isObj(params)) {
            throw new TypeError(`Expected object for params, but got ${params} (${typeof params})`)
        }
        const { messages, config } = params
        if (!Array.isArray(messages)) {
            throw new TypeError(`Expected array for messages, but got ${messages} (${typeof messages})`)
        }
        const pipelineInstance = await this.transformersPipelineFactory.getPipeline()

        // Async queue to bridge the callback-based streamer to an async generator
        const queue: string[] = []
        let done = false
        let error: unknown = null
        let resolveWait: (() => void) | null = null

        const streamer = new TextStreamer(pipelineInstance.tokenizer, {
            skip_prompt: true,
            callback_function(text: string) {
                queue.push(text)
                resolveWait?.()
            },
        })

        // Start generation (runs asynchronously while we yield from the queue)
        const normalizedMessages = normalizeMessageArray(messages)
        console.dir(normalizedMessages, {depth: null})
        const generatePromise = pipelineInstance(normalizedMessages, {
            ...defaultTextGenerationConfig,
            ...config,
            streamer,
        })
        
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