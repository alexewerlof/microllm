import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import { MicroEmbedder } from './MicroEmbedder.js'
import { MicroRAG } from './MicroRAG.js'
import { PipelineFactory } from './PipelineFactory.js'
import { VectorStore } from './VectorStore.js'

function createEmbedderWithEmbeddings(embeddingsByText: Record<string, number[]>): MicroEmbedder {
    const embedder = new MicroEmbedder(new PipelineFactory('feature-extraction', 'test-model'))
    embedder.embed = async (text: string) => {
        const embedding = embeddingsByText[text]

        if (!embedding) {
            throw new Error(`Missing test embedding for text: ${text}`)
        }

        return embedding
    }

    return embedder
}

describe(MicroRAG.name, () => {
    test('creates its own vector store when one is not provided', async () => {
        const chunk = '# Guide\nBody'
        const embedder = createEmbedderWithEmbeddings({
            [chunk]: [1, 0],
            Guide: [1, 0],
        })
        const rag = new MicroRAG(embedder)

        const indexedChunkCount = await rag.addDocument(chunk)
        const results = await rag.getRelevantDocuments('Guide', 0.5, 10)

        assert.strictEqual(indexedChunkCount, 1)
        assert.deepStrictEqual(results, [
            {
                text: chunk,
                metadata: {
                    title: 'Guide',
                    level: 1,
                    id: 1,
                    children: [],
                },
                score: 1,
            },
        ])
    })

    test('uses the injected vector store for indexing and retrieval', async () => {
        const chunk = '# Injected document\nBody'
        const embedder = createEmbedderWithEmbeddings({
            [chunk]: [0, 1],
            Injected: [0, 1],
        })
        const vectorStore = new VectorStore()
        const rag = new MicroRAG(embedder, vectorStore)

        const indexedChunkCount = await rag.addDocument(chunk, { source: 'custom-store' })
        const results = vectorStore.getSimilarRecords([0, 1], 0.5, 10)

        assert.strictEqual(indexedChunkCount, 1)
        assert.deepStrictEqual(results, [
            {
                text: chunk,
                metadata: {
                    source: 'custom-store',
                    title: 'Injected document',
                    level: 1,
                    id: 1,
                    children: [],
                },
                score: 1,
            },
        ])
    })

    test('throws when vectorStore is not a VectorStore instance', () => {
        const embedder = createEmbedderWithEmbeddings({})

        assert.throws(() => new MicroRAG(embedder, {} as never), {
            name: 'TypeError',
            message: 'Expected VectorStore instance for vectorStore, but got [object Object] (object)',
        })
    })
})
