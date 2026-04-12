import { isStr } from 'jty'
import {
    MicroChat,
    SupportedMessage,
    Tools,
    MicroAgent,
    isAssistantMessage,
    createUserMessage,
    PipelineFactory,
} from '../src/index.js'
import { createProgressCallback } from './progress-callback.js'

function getRandom() {
    const ret = Math.floor(Math.random() * 5)
    // Apparently the model has difficulty understanding scalar returns
    return `The random number is ${ret}.`
}

getRandom.description = 'Returns a random integer between 0 and 4.'

async function main() {
    const pipelineFactory = new PipelineFactory('text-generation', 'onnx-community/LFM2-1.2B-Tool-ONNX', {
        dtype: 'q4',
        progress_callback: createProgressCallback('Chat Pipeline'),
    })
    await pipelineFactory.getPipeline() // Pre-load the model before starting the agent loop
    const llm = new MicroChat(pipelineFactory)
    const messages: SupportedMessage[] = [
        {
            role: 'system',
            content: [
                `Use the provided "${getRandom.name}" function to generate a random integer between 0 and 4.`,
                `You should follow this process:`,
                `1. Call the function`,
                `2. Get the response and returns one of the following strings based on the response:`,
                `2.1. If the response is 0 return "STOP" and do **not** call the function again and end.`,
                `2.2. Otherwise, return the number you got and call the function again.`,
                `3. End when you get 0 and return "STOP".`,
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
        const resultMessages = await agent.work({ messages, tools })
        console.log(resultMessages)
        const lastMessage = resultMessages[resultMessages.length - 1]
        if (isAssistantMessage(lastMessage) && isStr(lastMessage.content) && lastMessage.content.toUpperCase().endsWith('STOP')) {
            break
        }
        messages.push(...resultMessages, createUserMessage('Call the function again.'))
        // eslint-disable-next-line no-constant-condition
    } while (true)
}

main().catch((error) => {
    console.error('An error occurred:', error)
    process.exit(1)
})
