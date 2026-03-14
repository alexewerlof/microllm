import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import { MicroLLM } from './MicroLLM'
import { SupportedMessage } from './Message/types'
import { Tools } from './Tools'

describe('MicroLLM', () => {
    test('uses tokenizer chat templating with tools when native support is available', async () => {
        const llm = new MicroLLM('test-model')
        const tools = new Tools()
        tools.addTool('get_time', 'Get the current time').func = async () => 'noon'

        const messages: SupportedMessage[] = [
            { role: 'system', content: 'You are a helpful assistant.' },
            { role: 'user', content: 'What time is it?' },
        ]

        const inputIds = [[1, 2, 3]]
        const outputIds = [[1, 2, 3, 4, 5]]
        let capturedMessages: unknown[] | undefined
        let capturedTemplateOptions: Record<string, unknown> | undefined

        llm.transformersPipelineFactory.getPipeline = async () => ({
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
        } as any)

        const result = await llm.complete({ messages, tools })

        assert.ok(capturedMessages)
        assert.ok(capturedTemplateOptions)
        assert.deepStrictEqual(capturedTemplateOptions.tools, tools.toJSON())
        assert.strictEqual(capturedTemplateOptions.add_generation_prompt, true)
        assert.strictEqual(
            capturedMessages.some((message: any) => message.role === 'system' && String(message.content).includes('List of tools:')),
            false,
        )
        assert.strictEqual(result.role, 'assistant')
        assert.ok('tool_calls' in result)
        assert.strictEqual(result.tool_calls[0].function.name, 'get_time')
    })

    test('logs the decoded prompt text with special tokens before generation', async () => {
        const llm = new MicroLLM('test-model')

        const messages: SupportedMessage[] = [
            { role: 'user', content: 'What time is it?' },
        ]

        const inputIds = [[1, 2, 3]]
        const outputIds = [[1, 2, 3, 4]]
        const loggedEntries: unknown[] = []
        const originalConsoleDir = console.dir

        llm.transformersPipelineFactory.getPipeline = async () => ({
            tokenizer: {
                apply_chat_template() {
                    return {
                        input_ids: inputIds,
                        attention_mask: [[1, 1, 1]],
                    }
                },
                batch_decode(tokenIds: unknown, options: Record<string, unknown> = {}) {
                    if (tokenIds === inputIds) {
                        if (options.skip_special_tokens) {
                            return ['user: What time is it?']
                        }

                        return ['<|system|>You are a helpful assistant.<|user|>What time is it?<|assistant|>']
                    }

                    return ['<|system|>You are a helpful assistant.<|user|>What time is it?<|assistant|>It is noon.']
                },
            },
            model: {
                async generate() {
                    return outputIds
                },
            },
        } as any)

        console.dir = (value: unknown) => {
            loggedEntries.push(value)
        }

        try {
            await llm.complete({ messages })
        } finally {
            console.dir = originalConsoleDir
        }

        assert.deepStrictEqual(loggedEntries[0], {
            promptTextWithSpecialTokens: '<|system|>You are a helpful assistant.<|user|>What time is it?<|assistant|>',
            inputs: {
                input_ids: inputIds,
                attention_mask: [[1, 1, 1]],
            },
        })
    })
})