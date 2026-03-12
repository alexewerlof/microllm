import { FunctionTool } from "../src/FunctionTool"
import { LiquidToolsLLM } from "../src/LiquidToolsLLM"
import { Tools } from "../src/Tools"

function getTime() {
    return String(new Date())
}

async function main() {
    const llm = new LiquidToolsLLM('onnx-community/LFM2-1.2B-Tool-ONNX', {
        dtype: 'q4',
    })
    const messages = [
        {
            role: 'system',
            content: 'You are a helpful assistant.'
        },
        {
            role: 'user',
            content: 'What time is it?'
        }
    ]

    const buff = []
    const tools = new Tools()
    tools.addTool('get_time', 'Get the current time').func = getTime
    for await (const token of llm.chatCompletion({
        messages,
        tools,
    })) {
        buff.push(token)
        process.stdout.write(token) // streams to terminal incrementally
    }
    console.log()
    console.log('Full response:', buff.join(''))
}

main().catch(console.error)
