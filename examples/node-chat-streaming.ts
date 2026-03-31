import { createUserMessage, SupportedMessage, MicroChat, PipelineFactory } from '../src/index.js'
import { createProgressCallback } from './progress-callback.js'

async function main() {
    const pipelineFactory = new PipelineFactory('text-generation', 'onnx-community/LFM2-1.2B-ONNX', {
        dtype: 'q4',
        progress_callback: createProgressCallback('Chat Pipeline'),
    })
    const llm = new MicroChat(pipelineFactory)

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
            const response = await llm.complete({
                messages,
                onToken: (token) => process.stdout.write(token),
            })
            console.log()
            messages.push(response)
        }
    } while (shouldContinue)

    process.stdin.destroy()
    console.log('Goodbye!')
}

main().catch((error) => {
    console.error('An error occurred:', error)
    process.exit(1)
})
