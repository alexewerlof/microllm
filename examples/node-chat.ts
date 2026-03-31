import { createUserMessage, SupportedMessage, MicroChat, PipelineFactory, Tools } from '../src/index.js'
import { createProgressCallback } from './progress-callback.js'

async function main() {
    const pipelineFactory = new PipelineFactory('text-generation', 'onnx-community/LFM2-1.2B-Tool-ONNX', {
        dtype: 'q4',
        progress_callback: createProgressCallback('Chat Pipeline'),
    })
    const llm = new MicroChat(pipelineFactory)

    const tools = new Tools()
    tools.addTool('get_time', 'Get the current time').func = async () => {
        return new Date().toLocaleTimeString()
    }

    const messages: SupportedMessage[] = [
        {
            role: 'system',
            content: 'You are a helpful assistant.',
        },
    ]

    let shouldContinue = true

    do {
        console.log('Prompt:')
        const userInput = await new Promise<string>((resolve) => {
            process.stdin.once('data', (data) => resolve(data.toString().trim()))
        })
        if (['exit', 'quit', ''].includes(userInput.toLowerCase())) {
            shouldContinue = false
        } else {
            messages.push(createUserMessage(userInput))
            console.log('Response:')
            // We pass the tools just to demonstrate what MicroChat emits. To call tools use MicroAgent.
            const assistantContent = await llm.complete({ messages, tools: tools.toJSON() })
            if (assistantContent.content) {
                console.log(assistantContent.content)
            } else {
                console.dir(assistantContent, { depth: null })
            }
            messages.push(assistantContent)
        }
    } while (shouldContinue)

    process.stdin.destroy()
    console.log('Goodbye!')
}

main().catch((error) => {
    console.error('An error occurred:', error)
    process.exit(1)
})
