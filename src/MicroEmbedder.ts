import { FeatureExtractionPipelineOptions } from '@huggingface/transformers'
import { TransformersPipelineFactory } from './TransformersPipelineFactory.js'

/**
 * Manages the embedding model lifecycle and generates embeddings.
 * Wraps the Transformers.js feature-extraction pipeline with lazy initialization.
 */
export class MicroEmbedder {
    #pipelineFactory: TransformersPipelineFactory<'feature-extraction'>

    /**
     * Creates a new Embedder instance.
     * @param modelId - Hugging Face model ID for embeddings.
     * @param pipelineOptions - Options passed directly to the Transformers.js pipeline (e.g. `{ dtype: "fp32" }`).
     */
    constructor(modelId: string, pipelineOptions: object = {}) {
        this.#pipelineFactory = new TransformersPipelineFactory('feature-extraction', modelId, pipelineOptions)
    }

    /**
     * Generates an embedding vector for the given text.
     * The pipeline must be initialized via `load()` before calling this method.
     * @param text - The input text to embed.
     * @returns The embedding vector.
     */
    async embed(text: string, options: FeatureExtractionPipelineOptions = {
        pooling: 'mean',
            normalize: true,
    }): Promise<number[]> {
        const pipelineInstance = await this.#pipelineFactory.getPipeline()
        const snippet = text.slice(0, 15)
        const logMsg = `Embed ${snippet}... (${text.length} chars)`
        console.time(logMsg)
        const output = await pipelineInstance(text, options)
        console.timeEnd(logMsg)

        return Array.from(output.data)
    }
}
