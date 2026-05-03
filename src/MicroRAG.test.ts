import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import { MicroEmbedder } from './MicroEmbedder.js'
import { MicroRAG } from './MicroRAG.js'
import { PipelineFactory } from './PipelineFactory.js'
import { VectorStore } from './VectorStore.js'
import { MicroLLM } from './MicroLLM.js'

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

    test('throws when complete is called without an llm delegate', async () => {
        const embedder = createEmbedderWithEmbeddings({})
        const rag = new MicroRAG(embedder)

        await assert.rejects(
            () => rag.complete({ messages: [{ role: 'user', content: 'Hello' }] }),
            /MicroRAG requires an llm delegate to use complete\(\)/,
        )
    })

    test('augments messages with similarity context and delegates completion to llm', async () => {
        const chunk = '# Guide\nAlways greet politely.'
        const embedder = createEmbedderWithEmbeddings({
            [chunk]: [1, 0],
            hello: [1, 0],
        })

        let capturedMessages: unknown[] = []
        const llm: MicroLLM<unknown> = {
            async complete(params) {
                capturedMessages = params.messages
                return { role: 'assistant', content: 'Hello there' }
            },
        }

        const rag = new MicroRAG(embedder, new VectorStore(), llm)
        await rag.addDocument(chunk)

        const result = await rag.complete({
            messages: [{ role: 'user', content: 'hello' }],
        })

        assert.strictEqual(result.role, 'assistant')
        assert.strictEqual(result.content, 'Hello there')
        assert.ok(capturedMessages.length >= 2)
        assert.strictEqual((capturedMessages[0] as { role?: string }).role, 'system')
    })
})
