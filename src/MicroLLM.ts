import { Message, PretrainedModelOptions, StoppingCriteriaList, TextGenerationConfig, TextStreamer } from "@huggingface/transformers"
import { TransformersPipelineFactory } from "./TransformersPipelineFactory"
import { hasProp, isA, isArr, isDef, isFn, isObj, isPOJO } from "jty"
import { normalizeMessageArray } from "./normalization"
import { SignalStoppingCriteria } from "./SignalStoppingCriteria"
import { Tools } from "./Tools"
import { SupportedMessage } from "./Message/types"
import { createSystemMessage } from "./Message/factories"
import { isSystemMessage } from "./Message/guards"
import { convertLiquidMessageToSupportedMessage, convertSupportedMessagesToLiquidMessages } from "./liquid-tools-transpiler"

const defaultTextGenerationConfig: Partial<TextGenerationConfig> = {
    max_new_tokens: 512,
    temperature: 0.5,
    top_p: 0.5,
}

export interface MicroLLMCompleteParams {
    /** The conversation messages. */
    messages: SupportedMessage[]
    /** Optional text generation config. */
    config?: Partial<TextGenerationConfig>
    /** Optional abort signal. */
    signal?: AbortSignal
    /** Optional tools instance whose declarations are injected into the prompt. */
    tools?: Tools
    /** Optional callback function which will receive every token as it's generated */
    onToken?: (token: string) => unknown
}

interface ToolCapablePipeline {
    tokenizer: {
        apply_chat_template: (...args: any[]) => any
        batch_decode: (...args: any[]) => string[]
    }
    model: {
        generate: (args: Record<string, unknown>) => Promise<unknown>
    }
}

/**
 * Adds the tool declarations as a synthetic system message for fallback pipelines
 * that do not expose native tool-aware chat templating.
 *
 * @param messages The supported conversation messages.
 * @param tools The available tool declarations.
 * @returns A cloned message array with the tool declaration message inserted.
 *
 * @example
 * ```ts
 * const prepared = injectToolDeclarations(messages, tools)
 * ```
 */
function injectToolDeclarations(messages: SupportedMessage[], tools: Tools): SupportedMessage[] {
    const supportedMessages = [...messages]
    const toolDeclarationsMessage = createSystemMessage('List of tools: ' + JSON.stringify(tools.toJSON()))
    const lastSystemIndex = supportedMessages.findLastIndex(isSystemMessage)
    supportedMessages.splice(lastSystemIndex + 1, 0, toolDeclarationsMessage)
    return supportedMessages
}

/**
 * Extracts the assistant continuation text from a decoded full-sequence generation.
 *
 * @param fullTexts The decoded generated sequences.
 * @param promptTexts The decoded prompt sequences.
 * @returns The generated assistant suffix.
 *
 * @example
 * ```ts
 * const text = extractAssistantText(['prompt answer'], ['prompt'])
 * ```
 */
function extractAssistantText(fullTexts: string[], promptTexts: string[]): string {
    if (!isArr(fullTexts) || !isArr(promptTexts) || typeof fullTexts[0] !== 'string' || typeof promptTexts[0] !== 'string') {
        throw new TypeError('Expected decoded prompt and output text arrays from tokenizer.batch_decode().')
    }

    return fullTexts[0].slice(promptTexts[0].length).trim()
}

export class MicroLLM {
    transformersPipelineFactory: TransformersPipelineFactory<"text-generation">
    
    constructor(modelId: string, pipelineOptions: PretrainedModelOptions = {}) {
        this.transformersPipelineFactory = new TransformersPipelineFactory('text-generation', modelId, pipelineOptions)
    }

    /**
     * Completes a conversation by sending messages to the model pipeline.
     * When `tools` is provided, tool declarations are injected as a system message
     * and tool call messages are converted to the Python format expected by Liquid AI models.
     *
     * @returns The raw assistant text from the model.
     *
     * @example
     * ```ts
     * const text = await llm.complete({ messages: [{ role: 'user', content: 'Hello' }] })
     *
     * // With tool definitions injected
     * const tools = new Tools()
     * tools.addTool('get_time', 'Get the current time').func = () => String(new Date())
     * const text = await llm.complete({ messages, tools })
     * ```
     */
    async complete(params: MicroLLMCompleteParams): Promise<SupportedMessage> {
        if (!isObj(params)) {
            throw new TypeError(`Expected object for params, but got ${params} (${typeof params})`)
        }
        const { signal, messages, config, tools, onToken } = params

        if (!Array.isArray(messages)) {
            throw new TypeError(`Expected array for messages, but got ${messages} (${typeof messages})`)
        }

        if (isDef(tools)) {
            if (!isA(tools, Tools)) {
                throw new TypeError(`Expected instance of Tools for tools, but got ${JSON.stringify(tools)} (${typeof tools})`)
            }
        }

        if (isDef(config)) {
            if (!isPOJO(config)) {
                throw new TypeError(`When specified, completion config should be an object. Got ${config} (${typeof config})`)
            }
        }

        const textGenerationConfig: Partial<TextGenerationConfig> = {
            ...defaultTextGenerationConfig,
            ...config,
        }

        const stoppingCriteriaConfig: {
            stopping_criteria?: StoppingCriteriaList,
        } = {}
        if (isDef(signal)) {
            if (signal.aborted) {
                throw new Error('The signal is already aborted. Cannot start generation.')
            }
            stoppingCriteriaConfig.stopping_criteria = SignalStoppingCriteria.createStoppingCriteriaList(signal)
        }

        const pipelineInstance = await this.transformersPipelineFactory.getPipeline()

        const streamerConfig: {
            streamer?: TextStreamer
        } = {}
        if (isDef(onToken)) {
            streamerConfig.streamer = new TextStreamer(pipelineInstance.tokenizer, {
                skip_prompt: true,
                callback_function: onToken,
            })
        }

        const supportedMessages = [...messages]

        const prepared: Message[] = convertSupportedMessagesToLiquidMessages(supportedMessages)

        const normalizedMessages = normalizeMessageArray(prepared)

        const inputs = pipelineInstance.tokenizer.apply_chat_template(normalizedMessages, {
            tools: tools?.toJSON(),
            add_generation_prompt: true,
            return_dict: true,
        }) as Record<string, unknown>
        const outputTokenIds = await pipelineInstance.model.generate({
            ...inputs,
            ...streamerConfig,
            ...stoppingCriteriaConfig,
            ...textGenerationConfig,
        })
        const decodedOutputs = pipelineInstance.tokenizer.batch_decode(outputTokenIds as any, {
            skip_special_tokens: true,
        })
        const decodedPrompts = pipelineInstance.tokenizer.batch_decode(inputs.input_ids as any, {
            skip_special_tokens: true,
        })

        return convertLiquidMessageToSupportedMessage({
            role: 'assistant',
            content: extractAssistantText(decodedOutputs, decodedPrompts),
        })
    }
}