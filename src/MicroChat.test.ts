import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import { MicroChat } from './MicroChat.js'
import { PipelineFactory } from './PipelineFactory.js'
import { SupportedMessage } from './Message/types.js'
import { FunctionToolDeclaration } from './Tools/types.js'
import { ChatModelAdapter } from './adapters/types.js'

describe('MicroChat', () => {
    test('throws when adapter does not expose required lifecycle hooks', () => {
        const factory = new PipelineFactory('text-generation', 'test-model')

        assert.throws(
            () => new MicroChat(factory, {} as ChatModelAdapter),
            /Expected adapter to expose onBeforeChatTemplate\(\) and onAfterDecode\(\)/,
        )
    })

    test('uses tokenizer chat templating with tools when native support is available', async () => {
        const factory = new PipelineFactory('text-generation', 'test-model')
        const llm = new MicroChat(factory)
        const tools: FunctionToolDeclaration[] = [
            {
                type: 'function',
                function: {
                    name: 'get_time',
                    description: 'Get the current time',
                    parameters: {
                        type: 'object',
                        properties: {},
                        required: [],
                        additionalProperties: false,
                    },
                },
            },
        ]

        const messages: SupportedMessage[] = [
            { role: 'system', content: 'You are a helpful assistant.' },
            { role: 'user', content: 'What time is it?' },
        ]

        const inputIds = [[1, 2, 3]]
        const outputIds = [[1, 2, 3, 4, 5]]
        let capturedMessages: unknown[] | undefined
        let capturedTemplateOptions: Record<string, unknown> | undefined

        llm.pipelineFactory.getPipeline = async () =>
            ({
                tokenizer: {
                    apply_chat_template(conversation: unknown[], options: Record<string, unknown>) {
                        capturedMessages = conversation
                        capturedTemplateOptions = options
                        return {
                            input_ids: inputIds,
                            attention_mask: [[1, 1, 1]],
                        }
                    },
                    batch_decode(tokenIds: unknown, options: Record<string, unknown> = {}) {
                        if (tokenIds === inputIds) {
                            return ['prompt']
                        }
                        if (options.skip_special_tokens) {
                            return ['prompt[get_time()]']
                        }
                        return ['prompt<|tool_call_start|>[get_time()]<|tool_call_end|>']
                    },
                },
                model: {
                    async generate() {
                        return outputIds
                    },
                },
            }) as any

        const result = await llm.complete({ messages, tools })

        assert.ok(capturedMessages)
        assert.ok(capturedTemplateOptions)
        assert.deepStrictEqual(capturedTemplateOptions.tools, tools)
        assert.strictEqual(capturedTemplateOptions.add_generation_prompt, true)
        assert.strictEqual(
            capturedMessages.some(
                (message: any) => message.role === 'system' && String(message.content).includes('List of tools:'),
            ),
            false,
        )
        assert.strictEqual(result.role, 'assistant')
        assert.ok('tool_calls' in result)
        assert.strictEqual(result.tool_calls[0].function.name, 'get_time')
    })

    test('throws when tools is not a function tool declaration array', async () => {
        const factory = new PipelineFactory('text-generation', 'test-model')
        const llm = new MicroChat(factory)

        await assert.rejects(
            () => llm.complete({ messages: [{ role: 'user', content: 'Hello' }], tools: {} as never }),
            {
                name: 'TypeError',
                message: 'Expected tools to be an array of FunctionToolDeclaration objects, but got {} (object)',
            },
        )
    })

    test('delegates lifecycle hooks to the injected adapter', async () => {
        const factory = new PipelineFactory('text-generation', 'test-model')

        const adapter: ChatModelAdapter = {
            name: 'test-adapter',
            onBeforeChatTemplate: ({ messages, tools }) => ({
                messages: messages as any,
                templateOptions: {
                    tools,
                    enable_thinking: true,
                },
            }),
            onAfterDecode: ({ cleanAssistantContent }) => ({
                role: 'assistant',
                content: `adapter:${cleanAssistantContent}`,
            }),
        }

        const llm = new MicroChat(factory, adapter)

        const messages: SupportedMessage[] = [{ role: 'user', content: 'Hello' }]
        const inputIds = [[1]]
        const outputIds = [[1, 2]]

        let capturedTemplateOptions: Record<string, unknown> | undefined

        llm.pipelineFactory.getPipeline = async () =>
            ({
                tokenizer: {
                    apply_chat_template(_conversation: unknown[], options: Record<string, unknown>) {
                        capturedTemplateOptions = options
                        return {
                            input_ids: inputIds,
                            attention_mask: [[1]],
                        }
                    },
                    batch_decode(tokenIds: unknown, options: Record<string, unknown> = {}) {
                        if (tokenIds === inputIds) {
                            return ['prompt']
                        }
                        if (options.skip_special_tokens) {
                            return ['promptclean']
                        }
                        return ['promptraw']
                    },
                },
                model: {
                    async generate() {
                        return outputIds
                    },
                },
            }) as any

        const result = await llm.complete({ messages })

        assert.ok(capturedTemplateOptions)
        assert.strictEqual(capturedTemplateOptions.enable_thinking, true)
        assert.strictEqual(result.role, 'assistant')
        assert.strictEqual(result.content, 'adapter:clean')
    })
})
