import { createAssistantMessage, createUserMessage } from "../src/Message/factories"
import { SupportedMessage } from "../src/Message/types"
import { MicroChat } from "../src/MicroChat"
import { PipelineFactory } from "../src/PipelineFactory"
import { createProgressCallback } from "./progress-callback"


async function main() {
    const pipelineFactory = new PipelineFactory('text-generation', 'onnx-community/LFM2-1.2B-Tool-ONNX', {
        dtype: 'q4',
        progress_callback: createProgressCallback('Chat Pipeline')
    })
    const llm = new MicroChat(pipelineFactory)

    const messages: SupportedMessage[] = [
        {
            role: 'system',
            content: 'You are a helpful assistant.'
        },
    ]

    let shouldContinue = true

    do {
        console.log('Prompt:')
        const userInput = await new Promise<string>(resolve => {
            process.stdin.once('data', data => resolve(data.toString().trim()))
        })
        if (['exit', 'quit', ''].includes(userInput.toLowerCase())) {
            shouldContinue = false
        } else {
            messages.push(createUserMessage(userInput))
            console.log('Response:')
            const assistantContent = await llm.complete({ messages })
            console.log(assistantContent.content)
            messages.push(assistantContent)
        }
    } while (shouldContinue)

    process.stdin.destroy()
    console.log('Goodbye!')
}

main().catch(error => {
    console.error('An error occurred:', error)
    process.exit(1)
})