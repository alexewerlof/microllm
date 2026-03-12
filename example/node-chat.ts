import { createUserMessage } from "../src/Message/factories"
import { MicroLLM } from "../src/MicroLLM"


async function main() {
    const llm = new MicroLLM('onnx-community/LFM2-1.2B-Tool-ONNX', {
        dtype: 'q4',
    })

    const messages = [
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
            const buff = []
            for await (const token of llm.chatCompletion({ messages })) {
                buff.push(token)
                process.stdout.write(token) // streams to terminal incrementally
            }
            console.log()
            messages.push({ role: 'assistant', content: buff.join('') })
        }
    } while (shouldContinue)

    process.stdin.destroy()
    console.log('Goodbye!')
}

main().catch(console.error)