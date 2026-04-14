import { JJD, JJHE } from 'jj'
import { PipelineFactory, MicroChat, createUserMessage } from 'micro-llm'

const jjDoc = JJD.from(document)
const status = jjDoc.find('#status')
const jjDevice = jjDoc.find('#device')
const jjModel = jjDoc.find('#model')
const jjLoadingProgress = jjDoc.find('#loading-progress')
const jjInitializeButton = jjDoc.find('#initialize-model')
const jjChatUI = jjDoc.find('#chatUI')
let pipelineFactory

async function hasUsableWebGpuAdapter() {
    if (!('gpu' in navigator) || typeof navigator.gpu?.requestAdapter !== 'function') {
        return false
    }

    return Boolean(await navigator.gpu.requestAdapter())
}

async function initializeDeviceOptions() {
    const isWebGpuUsable = await hasUsableWebGpuAdapter()
    if (isWebGpuUsable) {
        return
    }

    const webGpuOption = document.querySelector('#device option[value="webgpu"]')
    webGpuOption?.remove()

    if (jjDevice.getValue() === 'webgpu') {
        jjDevice.setValue('auto')
    }
}

function describeRuntimeError(error) {
    if (error instanceof Error) {
        if (error.message.includes('10290488')) {
            return 'Model initialization failed in this browser runtime. On Linux Chrome, enable the WebGPU flags from the README or try the WASM fallback on a machine with enough memory.'
        }

        return error.message
    }

    return String(error)
}

async function initialize() {
    await initializeDeviceOptions()
    const device = jjDevice.getValue()

    pipelineFactory = new PipelineFactory('text-generation', jjModel.getValue(), {
        dtype: 'q4',
        device,
        progress_callback: (progressInfo) => {
            switch (progressInfo.status) {
                case 'progress_total':
                    jjLoadingProgress.setValue(progressInfo.progress)
                    break
                case 'ready':
                    jjLoadingProgress.setValue(100)
                    break
            }
        },
    })
    return await pipelineFactory.getPipeline()
}

jjInitializeButton.on('click', async () => {
    try {
        await initialize()
        status.hide()
        jjChatUI.show()
    } catch (error) {
        console.error('Error initializing pipeline:', error)
        alert('Error initializing pipeline: ' + describeRuntimeError(error))
    }
})

const jjPrompt = jjDoc.find('#prompt').on('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault()
        void sendPrompt()
    }
})

jjDoc.find('#sendPrompt').on('click', () => {
    void sendPrompt()
})

const chatThread = jjDoc.find('#chatThread')

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

async function sendPrompt() {
    if (!pipelineFactory) {
        alert('Please initialize the model first')
        return
    }

    const userPrompt = jjPrompt.getValue()
    if (userPrompt.trim() === '') {
        alert('Please enter a prompt')
        return
    }
    jjPrompt.setValue('')
    const assistantMessage = JJHE.create('div')
    const latestChatResponse = JJHE.create('div').addChild(
        JJHE.create('h2').setText('User'),
        JJHE.create('div').setText(userPrompt),
        JJHE.create('h2').setText('Assistant'),
        assistantMessage,
    )

    chatThread.addChild(latestChatResponse)

    try {
        await chatCompletion(userPrompt, (token) => {
            const tokenElement = JJHE.create('span').setText(token)
            assistantMessage.addChild(tokenElement)
            console.log(token)
        })
    } catch (error) {
        console.error('Error during chat completion:', error)
        assistantMessage.setText(describeRuntimeError(error))
    }
}
