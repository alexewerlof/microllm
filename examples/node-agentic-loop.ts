import { MicroChat } from '../src/MicroChat'
import { SupportedMessage } from '../src/Message/types'
import { Tools } from '../src/Tools'
import { MicroAgent } from '../src/MicroAgent'
import { isAssistantMessage } from '../src/Message/guards'
import { createUserMessage } from '../src/Message/factories'
import { PipelineFactory } from '../src/PipelineFactory'
import { createProgressCallback } from './progress-callback'

function getRandom() {
    return Math.floor(Math.random() * 5)
}

getRandom.description = 'Returns a random integer between 0 and 4.'

async function main() {
    const pipelineFactory = new PipelineFactory('text-generation', 'onnx-community/LFM2-1.2B-Tool-ONNX', {
        dtype: 'q4',
        progress_callback: createProgressCallback('Chat Pipeline'),
    })
    const llm = new MicroChat(pipelineFactory)
    const messages: SupportedMessage[] = [
        {
            role: 'system',
            content: [
                `Use the provided "${getRandom.name}" function to generate a random integer between 0 and 4.`,
                `You should follow this process:`,
                `1. Call the function`,
                `2. If the result is equal to 0, just return STOP otherwise just return the result.`,
            ].join('\n'),
        },
        {
            role: 'user',
            content: [`Let me know when you get 3.`].join(' '),
        },
    ]

    const agent = new MicroAgent(llm)

    const tools = new Tools()
    tools.addTool(getRandom.name, getRandom.description).func = getRandom

    do {
        console.log('Current messages:', messages)
        const resultMessages = await agent.work(messages, tools)
        console.log(resultMessages)
        const lastMessage = resultMessages[resultMessages.length - 1]
        if (isAssistantMessage(lastMessage) && lastMessage.content.toUpperCase().endsWith('STOP')) {
            break
        }
        messages.push(...resultMessages, createUserMessage('Call the function again.'))
    } while (true)
}

main().catch((error) => {
    console.error('An error occurred:', error)
    process.exit(1)
})
