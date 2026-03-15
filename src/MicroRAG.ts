import { isStr, isA } from 'jty'
import { VectorStore } from './VectorStore.js'
import { MicroEmbedder } from './MicroEmbedder.js'

/**
 * Splits text into chunks by double-newline, keeping only non-empty chunks.
 * @param text - The raw text content.
 * @returns Non-empty chunks.
 */
export function chunkText(text: string): string[] {
    if (!isStr(text)) {
        throw new TypeError(`Expected string for text, but got ${text} (${typeof text})`)
    }
    return text
        .split(/\n\n+/)
        .map((c) => c.trim())
        .filter((c) => c.length)
}

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
     * @param vectorStore - Vector store for document storage and search.
     * @throws {TypeError} If embedder or vectorStore are not instances of their respective classes.
     */
    constructor(embedder: MicroEmbedder, vectorStore: VectorStore) {
        if (!isA(embedder, MicroEmbedder)) {
            throw new TypeError(`Expected MicroEmbedder instance for embedder, but got ${embedder} (${typeof embedder})`)
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
     * Chunks text, embeds each chunk, and adds them to the vector store.
     * @param text - The document text to add.
     * @param metadata - Optional metadata (e.g., { filename: "intro.md" }).
     * @returns The number of chunks indexed.
     */
    async addDocument(text: string, metadata: object = {}): Promise<number> {
        if (!isStr(text)) {
            throw new TypeError(`Expected string for text, but got ${text} (${typeof text})`)
        }

        const chunks = chunkText(text)
        for (const chunk of chunks) {
            const embedding = await this.#embedder.embed(chunk)
            this.#vectorStore.addDocument(chunk, embedding, metadata)
        }
        return chunks.length
    }

    /**
     * Retrieves the most relevant context chunks for a user query from the vector store.
     *
     * @param query - The user query to search for.
     * @param minScore - Minimum similarity score (0-1). See VectorStore.similarEmbeddings.
     * @param maxResults - Maximum number of results to return.
     * @returns An array of the most relevant results containing their score, text content, and structured metadata.
     */
    async getRelevantContext(
        query: string,
        minScore?: number,
        maxResults?: number,
    ): Promise<Array<{ score: number; text: string; metadata: object }>> {
        const queryEmbedding = await this.#embedder.embed(query)
        return this.#vectorStore.similarEmbeddings(queryEmbedding, minScore, maxResults)
    }

    /**
     * Augments a user query with retrieved context from the knowledge base.
     * @param query - The user query.
     * @param minScore - Minimum similarity score (0-1). See VectorStore.similarEmbeddings.
     * @param maxResults - Maximum number of results. See VectorStore.similarEmbeddings.
     * @returns The augmented prompt, or original query if no context found.
     */
    async augmentQuery(query: string, minScore?: number, maxResults?: number): Promise<string> {
        const context = await this.getRelevantContext(query, minScore, maxResults)
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
}
