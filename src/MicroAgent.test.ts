import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import { MicroChat } from './MicroChat.js'
import { MicroAgent } from './MicroAgent.js'
import { Tools } from './Tools.js'
import { SupportedMessage } from './Message/types.js'
import { PipelineFactory } from './PipelineFactory.js'
import { Message } from '@huggingface/transformers'

const MOCK_PROMPT_IDS = [[1, 2, 3]]
const MOCK_OUTPUT_IDS = [[1, 2, 3, 4, 5, 6]]
const MOCK_PROMPT_TEXT = 'prompt '

/**
 * Creates a mock pipeline object that returns canned responses in sequence.
 * Each call to model.generate() shifts the next response from the array.
 */
function createMockPipeline(responses: string[]) {
    let currentResponse = ''

    return {
        tokenizer: {
            apply_chat_template() {
                return { input_ids: MOCK_PROMPT_IDS, attention_mask: [[1, 1, 1]] }
            },
            batch_decode(tokenIds: unknown, options: Record<string, unknown> = {}) {
                if (tokenIds === MOCK_PROMPT_IDS) {
                    return [MOCK_PROMPT_TEXT]
                }
                if (options.skip_special_tokens) {
                    return [
                        MOCK_PROMPT_TEXT +
                            currentResponse.replaceAll('<|tool_call_start|>', '').replaceAll('<|tool_call_end|>', ''),
                    ]
                }
                return [MOCK_PROMPT_TEXT + currentResponse]
            },
        },
        model: {
            async generate() {
                currentResponse = responses.shift() ?? ''
                return MOCK_OUTPUT_IDS
            },
        },
    }
}

/**
 * Wires a MicroChat instance to use a mock pipeline with the given canned responses.
 */
function stubLLM(llm: MicroChat, responses: string[]) {
    const pipeline = createMockPipeline(responses)
    llm.pipelineFactory.getPipeline = async () => pipeline as any
}

describe('MicroAgent', () => {
    test('executes a model-requested tool and returns the follow-up answer', async () => {
        const llm = new MicroChat(new PipelineFactory('text-generation', 'test-model'))
        stubLLM(llm, ['<|tool_call_start|>[get_time()]<|tool_call_end|>', 'It is noon.'])

        let invoked = false
        const tools = new Tools()
        tools.addTool('get_time', 'Get the current time').func = async () => {
            invoked = true
            return 'noon'
        }

        const agent = new MicroAgent(llm)
        const messages: SupportedMessage[] = [{ role: 'user', content: 'What time is it?' }]
    const result = await agent.work({ messages, tools })

        assert.strictEqual(invoked, true)
        const lastMessage = result[result.length - 1]
        assert.strictEqual(lastMessage.role, 'assistant')
        assert.strictEqual(lastMessage.content, 'It is noon.')
    })

    test('returns text answer when response has no tool call tokens', async () => {
        const llm = new MicroChat(new PipelineFactory('text-generation', 'test-model'))
        stubLLM(llm, ['get_time()'])

        let invoked = false
        const tools = new Tools()
        tools.addTool('get_time', 'Get the current time').func = async () => {
            invoked = true
            return 'noon'
        }

        const agent = new MicroAgent(llm)
        const messages: SupportedMessage[] = [{ role: 'user', content: 'What time is it?' }]
    const result = await agent.work({ messages, tools })

        assert.strictEqual(invoked, false)
        const lastMessage = result[result.length - 1]
        assert.strictEqual(lastMessage.content, 'get_time()')
    })

    test('returns completion text when no tool calls are present', async () => {
        const llm = new MicroChat(new PipelineFactory('text-generation', 'test-model'))
        stubLLM(llm, ['final answer'])

        const tools = new Tools()
        const agent = new MicroAgent(llm)
        const messages: SupportedMessage[] = [{ role: 'user', content: 'Hello' }]
        const result = await agent.work({ messages, tools })

        const lastMessage = result[result.length - 1]
        assert.strictEqual(lastMessage.content, 'final answer')
    })

    test('strips tool call tokens from non-tool-call responses', async () => {
        const llm = new MicroChat(new PipelineFactory('text-generation', 'test-model'))
        stubLLM(llm, ['<|tool_call_start|>It is noon.<|tool_call_end|>'])

        const tools = new Tools()
        const agent = new MicroAgent(llm)
        const messages: SupportedMessage[] = [{ role: 'user', content: 'What time is it?' }]
        const result = await agent.work({ messages, tools })

        const lastMessage = result[result.length - 1]
        assert.strictEqual(lastMessage.content, 'It is noon.')
    })

    test('converts existing tool call messages to Python format before calling the pipeline', async () => {
        const llm = new MicroChat(new PipelineFactory('text-generation', 'test-model'))
        let capturedMessages: Message[] | undefined

        llm.pipelineFactory.getPipeline = async () =>
            ({
                tokenizer: {
                    apply_chat_template(messages: Message[]) {
                        capturedMessages = messages
                        return { input_ids: MOCK_PROMPT_IDS, attention_mask: [[1, 1, 1]] }
                    },
                    batch_decode(tokenIds: unknown) {
                        if (tokenIds === MOCK_PROMPT_IDS) {
                            return [MOCK_PROMPT_TEXT]
                        }
                        return [MOCK_PROMPT_TEXT + 'done']
                    },
                },
                model: {
                    async generate() {
                        return MOCK_OUTPUT_IDS
                    },
                },
            }) as any

        const tools = new Tools()
        tools.addTool('get_time', 'Get the current time').func = async () => 'noon'

        const agent = new MicroAgent(llm)
        const messages: SupportedMessage[] = [
            { role: 'user', content: 'What time is it?' },
            {
                role: 'assistant',
                tool_calls: [{ id: 'call_123', type: 'function', function: { name: 'get_time', arguments: '{}' } }],
            },
            { role: 'tool', tool_call_id: 'call_123', content: 'noon' },
        ]
        await agent.work({ messages, tools })

        assert.ok(capturedMessages)
        const assistantMsg = capturedMessages.find((m) => m.role === 'assistant')
        assert.strictEqual(assistantMsg?.content, '<|tool_call_start|>[get_time()]<|tool_call_end|>')
    })

    test('returns new array with generated messages', async () => {
        const llm = new MicroChat(new PipelineFactory('text-generation', 'test-model'))
        stubLLM(llm, ['hello'])

        const tools = new Tools()
        const agent = new MicroAgent(llm)
        const messages: SupportedMessage[] = [{ role: 'user', content: 'Hi' }]
        const result = await agent.work({ messages, tools })

        assert.notStrictEqual(result, messages)
        assert.strictEqual(result.length, 1)
        assert.strictEqual(result[0].role, 'assistant')
    })

    test('throws when maximum consecutive tool calls is exceeded', async () => {
        const llm = new MicroChat(new PipelineFactory('text-generation', 'test-model'))
        // Always return a tool call — should eventually exceed the limit
        const toolCallResponse = '<|tool_call_start|>[get_time()]<|tool_call_end|>'
        stubLLM(llm, Array(MicroAgent.MAX_TOOL_CALLS + 1).fill(toolCallResponse))

        const tools = new Tools()
        tools.addTool('get_time', 'Get the current time').func = async () => 'noon'

        const agent = new MicroAgent(llm)
        const messages: SupportedMessage[] = [{ role: 'user', content: 'What time is it?' }]

        await assert.rejects(() => agent.work({ messages, tools }), {
            message: `Maximum consecutive tool calls exceeded (${MicroAgent.MAX_TOOL_CALLS}).`,
        })
    })

    test('forwards the abort signal to the chat completion loop', async () => {
        const llm = new MicroChat(new PipelineFactory('text-generation', 'test-model'))
        let capturedGenerateArgs: Record<string, unknown> | undefined

        llm.pipelineFactory.getPipeline = async () =>
            ({
                tokenizer: {
                    apply_chat_template() {
                        return { input_ids: MOCK_PROMPT_IDS, attention_mask: [[1, 1, 1]] }
                    },
                    batch_decode(tokenIds: unknown) {
                        if (tokenIds === MOCK_PROMPT_IDS) {
                            return [MOCK_PROMPT_TEXT]
                        }
                        return [MOCK_PROMPT_TEXT + 'done']
                    },
                },
                model: {
                    async generate(args: Record<string, unknown>) {
                        capturedGenerateArgs = args
                        return MOCK_OUTPUT_IDS
                    },
                },
            }) as any

        const controller = new AbortController()
        const agent = new MicroAgent(llm)

        await agent.work({
            messages: [{ role: 'user', content: 'Hello' }],
            tools: new Tools(),
            signal: controller.signal,
        })

        assert.ok(capturedGenerateArgs)
        assert.ok('stopping_criteria' in capturedGenerateArgs)
    })
})
