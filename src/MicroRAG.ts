import { isStr, isA, isDef, isPOJO } from 'jty'
import { VectorStore, VectorStoreQueryResult } from './VectorStore.js'
import { MicroEmbedder } from './MicroEmbedder.js'
import { headerChunk } from './utilities/chunking.js'
import { createSystemMessage } from './Message/factories.js'

/**
 * Portable RAG (Retrieval-Augmented Generation) engine.
 * Works in both Node.js and browser environments.
 * Handles chunking, embedding, and context retrieval.
 */
export class MicroRAG {
    #embedder: MicroEmbedder
    #vectorStore: VectorStore

    /**
     * Initializes a new RAG instance.
     *
     * @param embedder - Initialized embedder instance used to convert text to vectors.
     * @param vectorStore - Optional store instance used for indexing and retrieval.
     * @throws {TypeError} If embedder is not an instance of MicroEmbedder.
     * @throws {TypeError} If vectorStore is not an instance of VectorStore.
     */
    constructor(embedder: MicroEmbedder, vectorStore: VectorStore = new VectorStore()) {
        if (!isA(embedder, MicroEmbedder)) {
            throw new TypeError(
                `Expected MicroEmbedder instance for embedder, but got ${embedder} (${typeof embedder})`,
            )
        }

        if (!isA(vectorStore, VectorStore)) {
            throw new TypeError(
                `Expected VectorStore instance for vectorStore, but got ${vectorStore} (${typeof vectorStore})`,
            )
        }

        this.#embedder = embedder
        this.#vectorStore = vectorStore
    }

    /**
     * Adds a document after chunking and calculating the embedding vectors.
     * @param text - The document text to add.
     * @param docMetadata - Optional metadata (e.g., { filename: "intro.md" }) to be added to all chunks.
     * @returns The number of chunks indexed.
     */
    async addDocument(text: string, docMetadata: object = {}): Promise<number> {
        if (!isStr(text)) {
            throw new TypeError(`Expected string for text, but got ${text} (${typeof text})`)
        }

        if (isDef(docMetadata) && !isPOJO(docMetadata)) {
            throw new TypeError(`Expected plain object for docMetadata, but got ${docMetadata} (${typeof docMetadata})`)
        }

        const chunks = headerChunk(text)
        for (const { content, metadata } of chunks) {
            const embedding = await this.#embedder.embed(content)
            this.#vectorStore.addRecord(content, embedding, { ...docMetadata, ...metadata })
        }
        return chunks.length
    }

    /**
     * Retrieves the most relevant context chunks for a user query from the vector store.
     *
     * @param query - The user query to search for.
     * @param minScore - Minimum similarity score (0-1). See VectorStore.getSimilarRecords.
     * @param maxResults - Maximum number of results to return.
     * @returns An array of the most relevant results containing their score, text content, and structured metadata.
     */
    async getRelevantDocuments(
        query: string,
        minScore?: number,
        maxResults?: number,
    ): Promise<VectorStoreQueryResult[]> {
        const queryEmbedding = await this.#embedder.embed(query)
        return this.#vectorStore.getSimilarRecords(queryEmbedding, minScore, maxResults)
    }

    async getSimilaritySystemMessage(query: string, minScore?: number, maxResults?: number) {
        const relevantContext = await this.getRelevantDocuments(query, minScore, maxResults)
        if (relevantContext.length === 0) {
            return undefined
        }
        // See the expected format here: https://docs.liquid.ai/lfm/models/lfm2-1.2b-rag
        const ragSystemPromptLines = [
            'The following documents may provide you additional information to answer questions:',
        ]
        for (let i = 0; i < relevantContext.length; i++) {
            const { text, metadata } = relevantContext[i]
            const tagName = `document${i + 1}`
            ragSystemPromptLines.push(`<${tagName} metadata=${JSON.stringify(metadata)}>\n${text}\n</${tagName}>`)
        }
        return createSystemMessage(ragSystemPromptLines.join('\n\n'))
    }
}
