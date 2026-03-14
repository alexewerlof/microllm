import { MicroLLM } from "../src/MicroLLM"
import { SupportedMessage } from "../src/Message/types"
import { Tools } from "../src/Tools"
import { MicroAgent } from "../src/MicroAgent"
import { isAssistantMessage } from "../src/Message/guards"
import { createUserMessage } from "../src/Message/factories"

function getRandom() {
    return Math.floor(Math.random() * 10)
}

getRandom.description = 'Returns a random integer between 0 and 9.'

async function main() {
    const llm = new MicroLLM('onnx-community/LFM2-1.2B-Tool-ONNX', {
        dtype: 'q4',
    })
    const messages: SupportedMessage[] = [
        {
            role: 'system',
            content: [
                `Use the provided "${getRandom.name}" function to generate a random integer between 0 and 9.`,
                `You should follow this process:`,
                `1. Call the function`,
                `2. Then do the following based on the result:`,
                `2.1. If the result is less than 5, your response should be RESULT < 5. Call the function again.`,
                `2.2. If the result is greater than 5, your response should be RESULT > 5. Call the function again.`,
                `2.3. If the result is exactly 5, your response should be RESULT = 5.`,
            ].join('\n')
        },
        {
            role: 'user',
            content: [
                `Let me know when you get 5.`
            ].join(' ')
        }
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
    } while(true)
}

main().catch(console.error)
