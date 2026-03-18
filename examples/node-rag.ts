import { glob } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { createSystemMessage, createUserMessage } from '../src/Message/factories'
import { SupportedMessage } from '../src/Message/types'
import { MicroChat } from '../src/MicroChat'
import { MicroEmbedder } from '../src/MicroEmbedder'
import { MicroRAG } from '../src/MicroRAG'
import { PipelineFactory } from '../src/PipelineFactory'
import { readFile } from 'node:fs/promises'
import { createProgressCallback } from './progress-callback'

async function loadContents(): Promise<{ filePath: string; content: string }[]> {
    const ret = []
    const searchRoot = resolve(import.meta.dirname)
    for await (const filePath of glob('content/**/*.md', { cwd: searchRoot })) {
        console.log(`Found file: ${filePath}`)
        const content = await readFile(join(searchRoot, filePath), 'utf-8')
        ret.push({ filePath, content })
    }
    return ret
}

async function main() {
    const embeddingPipelineFactory = new PipelineFactory('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
        dtype: 'q4',
        progress_callback: createProgressCallback('feature-extraction'),
    })

    const microEmbedder = new MicroEmbedder(embeddingPipelineFactory)
    await embeddingPipelineFactory.getPipeline()

    console.time('Initializing RAG...')
    const rag = new MicroRAG(microEmbedder)
    const fileContents = await loadContents()
    for (const { filePath, content } of fileContents) {
        console.time(filePath)
        await rag.addDocument(content, { src: filePath })
        console.timeEnd(filePath)
    }
    console.timeEnd('Initializing RAG...')

    // https://docs.liquid.ai/lfm/models/lfm2-1.2b-rag
    const chatPipelineFactory = new PipelineFactory('text-generation', 'onnx-community/LFM2-1.2B-RAG-ONNX', {
        dtype: 'q4',
        progress_callback: createProgressCallback('text-generation'),
    })

    const llm = new MicroChat(chatPipelineFactory)

    console.log('[Downloading and] loading pipelines...')

    await chatPipelineFactory.getPipeline()

    const originalSystemInstructions = 'You are an SRE expert'

    const messages: SupportedMessage[] = [createSystemMessage(originalSystemInstructions)]

    do {
        console.log('Prompt:')
        const userInput = await new Promise<string>((resolve) => {
            process.stdin.once('data', (data) => resolve(data.toString().trim()))
        })

        if (['exit', 'quit', ''].includes(userInput.trim().toLowerCase())) {
            break
        }

        messages.push(createUserMessage(userInput))
        const ephemeralMessages = [...messages]

        const relevantContext = await rag.getSimilaritySystemMessage(userInput, undefined, 3)
        if (relevantContext) {
            ephemeralMessages.push(relevantContext)
        }

        console.log('Response:')
        const response = await llm.complete({
            messages: ephemeralMessages,
            config: {
                max_new_tokens: 512,
                temperature: 0,
            },
            onToken: (token) => process.stdout.write(token),
        })
        console.log()
        messages.push(response)
        // eslint-disable-next-line no-constant-condition
    } while (true)

    process.stdin.destroy()
    console.log('Goodbye!')
}

main().catch((error) => {
    console.error('An error occurred:', error)
    process.exit(1)
})
