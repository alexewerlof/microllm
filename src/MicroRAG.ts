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
export class MicroRAG extends VectorStore{
    #embedder: MicroEmbedder

    /**
     * Initializes a new RAG instance.
     *
     * @param embedder - Initialized embedder instance used to convert text to vectors.
     * @param vectorStore - Vector store for document storage and search.
     * @throws {TypeError} If embedder or vectorStore are not instances of their respective classes.
     */
    constructor(embedder: MicroEmbedder) {
        if (!isA(embedder, MicroEmbedder)) {
            throw new TypeError(`Expected MicroEmbedder instance for embedder, but got ${embedder} (${typeof embedder})`)
        }
        super()
        this.#embedder = embedder
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
            this.addRecord(content, embedding, {...docMetadata, ...metadata})
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
        return this.getSimilarRecords(queryEmbedding, minScore, maxResults)
    }

    /**
     * Augments a user query with retrieved context from the knowledge base.
     * @param query - The user query.
     * @param minScore - Minimum similarity score (0-1). See VectorStore.getSimilarRecords.
     * @param maxResults - Maximum number of results. See VectorStore.getSimilarRecords.
     * @returns The augmented prompt, or original query if no context found.
     */
    async augmentQuery(query: string, minScore?: number, maxResults?: number): Promise<string> {
        const context = await this.getRelevantDocuments(query, minScore, maxResults)
        if (context.length === 0) {
            console.log('No RAG context found for query')
            return query
        }

        console.log(
            `Found ${context.length} RAG context for query. Similarity scores: ${context.map((r) => r.score.toFixed(3)).join(', ')}`,
        )
        return [
            'Background information context to help answer the user:',
            ...context.map((r) => r.text),
            '',
            'User Question:',
            query,
        ].join('\n')
    }

    async getSimilaritySystemMessage(query: string, minScore?: number, maxResults?: number) {
        const relevantContext = await this.getRelevantDocuments(query, minScore, maxResults)
        if (relevantContext.length === 0) {
            return undefined
        }
        // See the expected format here: https://docs.liquid.ai/lfm/models/lfm2-1.2b-rag
        const ragSystemPrompLines = [
            'The following documents may provide you additional information to answer questions:',
        ]
        for (let i = 0; i < relevantContext.length; i++) {
            const { text, metadata } = relevantContext[i]
            const tagName = `document${i + 1}`
            ragSystemPrompLines.push(`<${tagName} metadata=${JSON.stringify((metadata))}>\n${text}\n</${tagName}>`)
        }
        return createSystemMessage(ragSystemPrompLines.join('\n\n'))
    }
}
