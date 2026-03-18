import * as jj from 'jj'
import { PipelineFactory, MicroChat, createUserMessage } from 'micro-llm'

const status = jj.doc.find('#status')

const progressBars = new Map()

function getProgressIndicator(file) {
    if (!progressBars.has(file)) {
        const progressBar = jj.JJHE.create('progress').setAttr({
            min: 0,
            max: 100
        }).setValue(0)
        const container = jj.JJHE.create('div').addChild(
            jj.JJHE.create('h3').setText(file),
            progressBar,
        )
        status.addChild(container)
        progressBars.set(file, progressBar)
    }
    return progressBars.get(file)
}

const pipelineFactory = new PipelineFactory('text-generation', 'onnx-community/LFM2-1.2B-Tool-ONNX', {
    dtype: 'q4',
    progress_callback: (progressInfo) => {
        if (progressInfo.status !== 'progress') {
            return
        }
        const bar = getProgressIndicator(progressInfo.file)
        bar.setValue(progressInfo.progress)
        console.log(progressInfo.file, progressInfo.progress)
    },
})


jj.doc.find('#initializeModel').on('click', async () => {
    try {
        await pipelineFactory.getPipeline()
        console.log('Pipeline initialized')
        status.hide()
        jj.doc.find('#chatUI').show()
    } catch (error) {
        console.error('Error initializing pipeline:', error)
    }
})

const prompt = jj.doc.find('#prompt').on('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault()
        sendPrompt()
    }
})

jj.doc.find('#sendPrompt').on('click', sendPrompt)

const chatThread = jj.doc.find('#chatThread')

async function chatCompletion(userInput, onToken) {
    if (typeof onToken !== 'function') {
        throw new Error('onToken must be a function')
    }
    const llm = new MicroChat(pipelineFactory)

    const messages = [
        {
            role: 'system',
            content: 'You are a helpful assistant.',
        },
    ]

    console.log('Prompt:')
    messages.push(createUserMessage(userInput))
    console.log('Response:')
    const assistantContent = await llm.complete({ messages, onToken })
    console.log(assistantContent.content)
    messages.push(assistantContent)
}

function sendPrompt() {
    const userPrompt = prompt.getValue()
    if (userPrompt.trim() === '') {
        alert('Please enter a prompt')
        return
    }
    prompt.setValue('')
    const assistantMessage = jj.JJHE.create('div')
    const latestChatResponse = jj.JJHE.create('div').addChild(
        jj.JJHE.create('h2').setText('User'),
        jj.JJHE.create('div').setText(userPrompt),
        jj.JJHE.create('h2').setText('Assistant'),
        assistantMessage,
    )

    chatThread.addChild(latestChatResponse)

    chatCompletion(userPrompt, (token) => {
        const tokenElement = jj.JJHE.create('span').setText(token)
        assistantMessage.addChild(tokenElement)
        console.log(token)
    })
}