import { createUserMessage } from "../src/Message/factories"
import { SupportedMessage } from "../src/Message/types"
import { MicroChat } from "../src/MicroChat"


async function main() {
    const llm = new MicroChat('onnx-community/LFM2-1.2B-Tool-ONNX', {
        dtype: 'q4',
    })

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

main().catch(console.error)