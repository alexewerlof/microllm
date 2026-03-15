import { inRange, isArr, isArrLen, isInt, isPOJO } from 'jty'

/**
 * Calculates cosine similarity between two vectors.
 * Note: This function assumes that both vectors are already L2 normalized.
 * When using transformers.js, ensure you pass \`normalize: true\` to the pipeline options.
 * Because the vectors are normalized, this avoids calculating norms and uses a pure dot product.
 * @param v1 - First vector.
 * @param v2 - Second vector.
 * @returns Similarity score (0-1).
 */
export function cosineSimilarity(v1: number[], v2: number[]): number {
    if (v1.length !== v2.length) {
        throw new Error(`Vector length mismatch: ${v1.length} vs ${v2.length}`)
    }

    let dotProduct = 0

    for (let i = 0; i < v1.length; i++) {
        dotProduct += v1[i] * v2[i]
    }

    return dotProduct
}

/**
 * Simple in-memory vector database.
 * Portable between Node.js and Browser.
 */
export class VectorStore {
    /** Minimum length of the embedding vectors. */
    static MIN_EMBEDDING_LENGTH = 100
    /**
     * Length of the embedding vectors.
     * This value is initialized when the first document is added.
     * After that, it will be used to validate the dimension of the new documents or query embedding.
     */
    #embeddingDimension = 0
    #documents: Map<string, { embedding: number[]; metadata: object }>

    constructor() {
        this.#documents = new Map()
    }

    /**
     * Adds a document and its embedding to the store.
     * @param {string} text - The original text.
     * @param {number[]} embedding - The vector embedding.
     * @param {object} [metadata={}] - Optional metadata (e.g., filename).
     * @returns {boolean} True if added, false if it already exists.
     */
    addDocument(text: string, embedding: number[], metadata: object = {}): boolean {
        if (!isArr(embedding)) {
            throw new TypeError(`Expected embedding to be an array, got ${embedding} (${typeof embedding})`)
        }
        if (!isArrLen(embedding, VectorStore.MIN_EMBEDDING_LENGTH)) {
            throw new RangeError(`Expected embedding to have at least ${VectorStore.MIN_EMBEDDING_LENGTH} elements, got ${(embedding as number[])?.length}`)
        }
        if (this.#documents.has(text)) {
            return false
        }
        if (this.#embeddingDimension === 0) {
            this.#embeddingDimension = embedding.length
        } else if (this.#embeddingDimension !== embedding.length) {
            throw new RangeError(
                `Expected embedding dimension to be ${this.#embeddingDimension}, got ${embedding.length}`,
            )
        }
        this.#documents.set(text, { embedding, metadata })
        return true
    }

    /**
     * Searches for the most similar documents.
     * @param queryEmbedding - The embedding of the query.
     * @param minScore - Minimum similarity score (0-1) required for a result to be included.
     * @param maxResults - Maximum number of results to return.
     * @returns Array of objects containing text, metadata, and similarity score.
     */
    similarEmbeddings(
        queryEmbedding: number[],
        minScore = 0.3,
        maxResults = 0,
    ): Array<{ text: string; metadata: object; score: number }> {
        if (!isArr(queryEmbedding)) {
            throw new TypeError(
                `Expected queryEmbedding to be an array, got ${queryEmbedding} (${typeof queryEmbedding})`,
            )
        }
        if (!isArrLen(queryEmbedding, VectorStore.MIN_EMBEDDING_LENGTH)) {
            throw new RangeError(`Expected queryEmbedding to have at least ${VectorStore.MIN_EMBEDDING_LENGTH} elements, got ${(queryEmbedding as number[])?.length}`)
        }
        if (!inRange(minScore, 0, 1)) {
            throw new RangeError(`Expected minScore to be a number between 0 and 1, got ${minScore}`)
        }
        if (!isInt(maxResults) || maxResults < 0) {
            throw new RangeError(`Expected maxResults to be 0 or a positive integer, got ${maxResults}`)
        }
        if (this.#embeddingDimension && this.#embeddingDimension !== queryEmbedding.length) {
            throw new RangeError(
                `Expected embedding dimension to be ${this.#embeddingDimension}, got ${queryEmbedding.length}`,
            )
        }
        const results = []
        for (const [text, { embedding, metadata }] of this.#documents.entries()) {
            const score = cosineSimilarity(queryEmbedding, embedding)
            if (score >= minScore) {
                results.push({ text, metadata, score })
            }
        }

        results.sort((a, b) => b.score - a.score)
        if (maxResults > 0 && results.length > maxResults) {
            results.length = maxResults
        }

        return results
    }

    toJSON(): Record<string, { embedding: number[]; metadata: object }> {
        return Object.fromEntries(this.#documents.entries())
    }

    static fromJSON(json: Record<string, { embedding: number[]; metadata: object }>): VectorStore {
        if (!isPOJO(json)) {
            throw new TypeError(`Expected a plain object to deserialize, got ${json} (${typeof json})`)
        }
        const store = new VectorStore()
        for (const [text, { embedding, metadata }] of Object.entries(json)) {
            store.addDocument(text, embedding, metadata)
        }
        return store
    }
}
