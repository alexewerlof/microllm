import { input } from '@inquirer/prompts'
import { createUserMessage, SupportedMessage, MicroChat, LiquidAdapter } from '../src/index.js'
import { inquirePipeline } from './pipeline-selection.js'

async function main() {
    const pipelineFactory = await inquirePipeline()
    const llm = new MicroChat(pipelineFactory, new LiquidAdapter())

    await pipelineFactory.getPipeline()

    const messages: SupportedMessage[] = [
        {
            role: 'system',
            content: 'You are a helpful assistant.',
        },
    ]

    let shouldContinue = true

    do {
        console.log()
        const userInput = await input({
            message: 'Prompt:',
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

    console.log('Goodbye!')
}

main().catch((error) => {
    console.error('An error occurred:', error)
    process.exit(1)
})
