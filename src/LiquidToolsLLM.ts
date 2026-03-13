import { Message, PretrainedModelOptions } from '@huggingface/transformers'
import { MicroLLM } from './MicroLLM'
import { createToolCallsMessage, createToolResultMessage } from './Message/factories'
import { SupportedMessage, ToolCallObj } from './Message/types'
import { FunctionTool, FunctionToolDeclaration, FunctionToolFunctionDeclaration } from './FunctionTool'
import { convertToolCallsMsgToPython, parsePythonToolCallObj } from './liquid-tools-transpiler'
import { normalizeMessageArray } from './normalization'
import { Tools } from './Tools'
import { isA, isDef } from 'jty'

function stringifyToolResult(result: unknown): string {
    if (typeof result === 'string') {
        return result
    }
    return JSON.stringify(result)
}

export class LiquidToolsLLM extends MicroLLM {
    constructor(modelId: string, pipelineOptions: PretrainedModelOptions = {}) {
        super(modelId, pipelineOptions)
    }

    async #generateText(messages: SupportedMessage[], tools: FunctionToolDeclaration[] = []): Promise<string> {
        const pipelineInstance: any = await this.transformersPipelineFactory.getPipeline()
        const normalizedMessages = normalizeMessageArray(convertToolCallsMsgToPython(messages))
        const inputs = pipelineInstance.tokenizer.apply_chat_template(normalizedMessages, {
            add_generation_prompt: true,
            return_dict: true,
            ...(tools.length > 0 ? { tools } : {}),
        })
        const outputTokenIds = await pipelineInstance.model.generate({
            ...inputs,
            max_new_tokens: 128,
        })
        const decoded = pipelineInstance.tokenizer.batch_decode(outputTokenIds, {
            skip_special_tokens: true,
        })
        const promptTexts = pipelineInstance.tokenizer.batch_decode(inputs.input_ids, {
            skip_special_tokens: true,
        })

        return decoded[0].slice(promptTexts[0].length)
    }

    async *generateTokens(params: {
        messages: Message[]
        tools: Tools
    }): AsyncGenerator<string, void, unknown> {
        const { messages, tools } = params
        if (!isA(tools, Tools)) {
            throw new TypeError(`Expected tools to be an instance of Tools. Got ${tools} (${typeof tools})`)
        }
        let conversation = params.messages as SupportedMessage[]

        for (let iteration = 0; iteration < 4; iteration++) {
            const assistantText = await this.#generateText(conversation, tools.toJSON())
            let toolCalls: ToolCallObj[] | null = null

            if (tools.length > 0) {
                try {
                    toolCalls = parsePythonToolCallObj(assistantText)
                } catch {
                    toolCalls = null
                }
            }

            if (!toolCalls?.length) {
                if (assistantText.length > 0) {
                    yield assistantText
                }
                return
            }

            conversation = [...conversation, createToolCallsMessage(toolCalls)]
            const toolResults = await tools.exeToolCalls({ tool_calls: toolCalls })
            conversation.push(...toolResults.map((result) => createToolResultMessage(result.tool_call_id, result.content)))
        }

        throw new Error('Exceeded maximum number of tool-calling iterations.')
    }
}