import { MicroAPI } from '../src/index.js'

const microApi = new MicroAPI({
    baseUrl: 'http://192.168.1.17:1234/v1',
    modelId: 'liquid/lfm2.5-1.2b',
    apiKey: 'test',
})

async function test() {
    const response = await microApi.complete({
        messages: [
            {
                role: 'user',
                content: 'What is the meaning of life?',
            },
        ],
        config: {
            temperature: 0.5,
        },
    })

    console.log('Response:', response)
}

test().catch(console.error)
