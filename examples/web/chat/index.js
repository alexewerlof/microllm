import { JJD, JJHE } from 'jj'
import { PipelineFactory, MicroChat, createUserMessage, createSystemMessage } from 'micro-llm'

const h = JJHE.tree

const jjDoc = JJD.from(document)
const jjInitControls = jjDoc.find('#init-controls', true)
const jjLoadProg = jjDoc.find('#loading-progress', true)
const jjDevice = jjDoc.find('#device', true)
const jjChatUI = jjDoc.find('#chatUI', true)
const jjModel = jjDoc.find('#model', true)
const jjSystemPrompt = jjDoc.find('#system-prompt', true)
const jjPrompt = jjDoc.find('#prompt', true).on('keydown', handlePromptKeydown)
const jjChatThread = jjDoc.find('#chat-thread', true)

let pipelineFactory

const messages = []

jjDoc.find('#initialize-model', true).on('click', async () => {
    try {
        pipelineFactory = new PipelineFactory('text-generation', jjModel.getValue(), {
            dtype: 'q4',
            device: jjDevice.getValue(),
            progress_callback: (progressInfo) => {
                switch (progressInfo.status) {
                    case 'progress_total':
                        jjLoadProg.setValue(progressInfo.progress)
                        break
                    case 'ready':
                        jjLoadProg.setValue(100)
                        break
                }
            },
        })
        await pipelineFactory.getPipeline()

        jjInitControls.swAttr('disabled', true)
        jjChatUI.swAttr('hidden', false)
        messages.push(createSystemMessage(jjSystemPrompt.getValue()))
    } catch (error) {
        console.error('Error initializing pipeline:', error)
        alert(`Error initializing pipeline: ${error}`)
    }
})

function handlePromptKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey && !event.ctrlKey) {
        event.preventDefault()
        void sendPrompt()
    }
}

async function chatCompletion(userInput, onToken) {
    if (typeof onToken !== 'function') {
        throw new Error('onToken must be a function')
    }
    if (!pipelineFactory) {
        throw new ReferenceError('Pipeline factory is not initialized')
    }
    const llm = new MicroChat(pipelineFactory)

    console.log('Prompt:')
    messages.push(createUserMessage(userInput))
    console.log('Response:')
    const maxNewTokens = parseInt(jjDoc.find('#max-new-tokens', true).getValue(), 10) || 512
    console.debug({ maxNewTokens })
    const assistantMessage = await llm.complete({ messages, onToken, config: {
        max_new_tokens: maxNewTokens,
    }})
    console.log(assistantMessage)
    messages.push(assistantMessage)
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

    const jjAssistantMessage = h('div')
    const jjChatResponse = h('div', null,
        h('h2', null, 'User'),
        h('div', null, userPrompt),
        h('h2', null, 'Assistant'),
        jjAssistantMessage,
    )

    jjChatThread.addChild(jjChatResponse)

    try {
        await chatCompletion(userPrompt, (token) => {
            const span = h('span')
            span.ref.innerText = token
            jjAssistantMessage.addChild(span)
            console.log(token)
        })
    } catch (error) {
        console.error('Error during chat completion:', error)
    }
}

jjDoc.find('#sendPrompt', true).on('click', () => void sendPrompt())

async function hasUsableWebGpuAdapter() {
    if (typeof navigator?.gpu?.requestAdapter !== 'function') {
        return false
    }

    return Boolean(await navigator.gpu.requestAdapter())
}

async function initializeDeviceOptions() {
    const isWebGpuUsable = await hasUsableWebGpuAdapter()
    if (isWebGpuUsable) {
        // Select it by default
        jjDevice.setValue('webgpu')
    } else {
        jjDevice.setValue('auto').find('option[value="webgpu"]')?.rm()
    }
}

async function main() {
    await initializeDeviceOptions()
}

main().catch((error) => {
    console.error('Error in main:', error)
    alert(`Error: ${error}`)
})