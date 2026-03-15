import { MicroChat } from "../src/MicroChat"
import { SupportedMessage } from "../src/Message/types"
import { Tools } from "../src/Tools"
import { MicroAgent } from "../src/MicroAgent"

function getTime() {
    console.log('------- inside getTime() -------')
    return String(new Date())
}

async function main() {
    const llm = new MicroChat('onnx-community/LFM2-1.2B-Tool-ONNX', {
        dtype: 'q4',
    })
    const messages: SupportedMessage[] = [
        {
            role: 'system',
            content: 'You are a helpful assistant.'
        },
        {
            role: 'user',
            content: 'Use the get_time function to tell me what time it is.'
        }
    ]

    const agent = new MicroAgent(llm)

    
    const tools = new Tools()
    tools.addTool('get_time', 'Get the current time in the current location').func = getTime
    const resultMessages = await agent.work(messages, tools)
    const lastMessage = resultMessages[resultMessages.length - 1]
    console.log('Full response:', lastMessage.content)
}

main().catch(console.error)
