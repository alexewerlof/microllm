import { FeatureExtractionPipelineOptions } from '@huggingface/transformers'
import { TransformersPipelineFactory } from './TransformersPipelineFactory.js'

/**
 * Generates embeddings using a caller-owned feature-extraction pipeline factory.
 */
export class MicroEmbedder {
    pipelineFactory: TransformersPipelineFactory<'feature-extraction'>

    /**
     * Creates a new Embedder instance.
     * @param pipelineFactory - Factory used to resolve the feature-extraction pipeline.
     *
     * @example
     * ```ts
     * const factory = new TransformersPipelineFactory('feature-extraction', 'Xenova/all-MiniLM-L6-v2')
     * const embedder = new MicroEmbedder(factory)
     * ```
     */
    constructor(pipelineFactory: TransformersPipelineFactory<'feature-extraction'>) {
        this.pipelineFactory = pipelineFactory
    }

    /**
     * Generates an embedding vector for the given text.
     * @param text - The input text to embed.
     * @returns The embedding vector.
     */
    async embed(text: string, options: FeatureExtractionPipelineOptions = {
        pooling: 'mean',
            normalize: true,
    }): Promise<number[]> {
        const pipelineInstance = await this.pipelineFactory.getPipeline()
        const snippet = text.slice(0, 15)
        const logMsg = `Embed ${snippet}... (${text.length} chars)`
        console.time(logMsg)
        const output = await pipelineInstance(text, options)
        console.timeEnd(logMsg)

        return Array.from(output.data)
    }
}
