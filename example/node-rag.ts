import { glob } from "node:fs/promises"
import { join, resolve } from "node:path"
import { createSystemMessage, createUserMessage } from "../src/Message/factories"
import { SupportedMessage, SystemMessage } from "../src/Message/types"
import { MicroChat } from "../src/MicroChat"
import { MicroEmbedder } from "../src/MicroEmbedder"
import { MicroRAG } from "../src/MicroRAG"
import { PipelineFactory } from "../src/PipelineFactory"
import { VectorStore } from "../src/VectorStore"
import { readFile } from 'node:fs/promises'
import { headerChunk } from "../src/chunking"

async function createVectorDbFromContent(embedder: MicroEmbedder, contentRoot: string): Promise<VectorStore> {
    const vectorStore = new VectorStore()
    for await (const filePath of glob('**/*.md', { cwd: contentRoot })) {
        console.log(`Found file: ${filePath}`)
        const document = await readFile(join(contentRoot, filePath), 'utf-8')
        const chunks = headerChunk(document, { docId: filePath })
        for (const chunk of chunks) {
            const embedding = await embedder.embed(chunk.content)
            vectorStore.addDocument(chunk.content, embedding, { filename: filePath })
        }
    }
    return vectorStore
}
async function main() {
    const embeddingPipelineFactory = new PipelineFactory('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'q4' })
    // https://docs.liquid.ai/lfm/models/lfm2-1.2b-rag
    const chatPipelineFactory = new PipelineFactory('text-generation', 'onnx-community/LFM2-1.2B-RAG-ONNX', { dtype: 'q4'})

    const microEmbedder = new MicroEmbedder(embeddingPipelineFactory)
    await embeddingPipelineFactory.getPipeline()

    console.time('Initializing vector store...')
    const vectorStore = await createVectorDbFromContent(microEmbedder, resolve(import.meta.dirname, 'content'))
    console.timeEnd('Initializing vector store...')

    const rag = new MicroRAG(microEmbedder, vectorStore)
    const llm = new MicroChat(chatPipelineFactory)

    console.log('[Downloading and] loading pipelines...')

    await chatPipelineFactory.getPipeline()

    const originalSystemInstructions = 'You are an SRE expert'
    
    do {
        console.log('Prompt:')
        const userInput = await new Promise<string>(resolve => {
            process.stdin.once('data', data => resolve(data.toString().trim()))
        })
 
        if (['exit', 'quit', ''].includes(userInput.trim().toLowerCase())) {
            break
        }

        const systemPrompt = createSystemMessage(originalSystemInstructions)

        const relevantContext = await rag.getRelevantContext(userInput, undefined, 3)
        if (relevantContext.length > 0) {
            const ragSystemPromp = [originalSystemInstructions]
            for (let i = 0; i < relevantContext.length; i++) {
                const { score, text, metadata } = relevantContext[i]
                const tagName = `document${i + 1}`
                ragSystemPromp.push(`<${tagName} metadata=${JSON.stringify((metadata))}>\n${text}\n</${tagName}>`)
            }
            systemPrompt.content = ragSystemPromp.join('\n\n')
        }

        const messages: SupportedMessage[] = [
            systemPrompt,
            createUserMessage(userInput),
        ]
        console.log('Response:')
        const response = await llm.complete({
            messages,
            config: {
                max_new_tokens: 512,
                temperature: 0,
            },
            onToken: (token) => process.stdout.write(token),
        })
        console.log()
        messages.push(response)
    } while (true)

    process.stdin.destroy()
    console.log('Goodbye!')
}

main().catch(console.error)