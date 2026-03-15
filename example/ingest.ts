import { glob, mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { MicroEmbedder } from '../src/MicroEmbedder'
import { TransformersPipelineFactory } from '../src/TransformersPipelineFactory.js'
import { chunkText } from '../src/RAG.js'
import { VectorStore } from '../src/VectorStore'

const embedderPipelineFactory = new TransformersPipelineFactory('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'q4' })
const microEmbedder = new MicroEmbedder(embedderPipelineFactory)

const vectorStore = new VectorStore()
const contentDir = new URL('./content/', import.meta.url)
const outputFile = new URL('../.cache/db.json', import.meta.url)

for await (const filePath of glob('**/*.md', { cwd: contentDir })) {
    console.log(`Found file: ${filePath}`)
    const document = await readFile(new URL(filePath, contentDir), 'utf-8')
    const chunks = chunkText(document)
    for (const chunk of chunks) {
        const embedding = await microEmbedder.embed(chunk)
        vectorStore.addDocument(chunk, embedding, { filename: filePath })
    }
}

await mkdir(dirname(fileURLToPath(outputFile)), { recursive: true })
await writeFile(outputFile, JSON.stringify(vectorStore.toJSON(), null, 2))
