import { Message, PretrainedModelOptions, StoppingCriteriaList, Tensor, TextGenerationConfig, TextStreamer } from "@huggingface/transformers"
import { TransformersPipelineFactory } from "./TransformersPipelineFactory"
import { hasProp, isA, isArr, isDef, isFn, isObj, isPOJO, isStr } from "jty"
import { normalizeMessageArray } from "./normalization"
import { SignalStoppingCriteria } from "./SignalStoppingCriteria"
import { Tools } from "./Tools"
import { SupportedMessage } from "./Message/types"
import { createAssistantMessage } from "./Message/factories"
import { convertSupportedMessagesToLiquidMessages, tryParseToolCalls } from "./liquid-tools-transpiler"

const defaultTextGenerationConfig: Partial<TextGenerationConfig> = {
    max_new_tokens: 512,
    temperature: 0.5,
    top_p: 0.5,
}

export interface MicroChatCompleteParams {
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
    if (!isArr(fullTexts) || !isArr(promptTexts) || !isStr(fullTexts[0]) || !isStr(promptTexts[0])) {
        throw new TypeError('Expected decoded prompt and output text arrays from tokenizer.batch_decode().')
    }

    return fullTexts[0].slice(promptTexts[0].length).trim()
}

/**
 * Decodes tokenized model output and prompt, then extracts the assistant continuation.
 *
 * @param tokenizer The tokenizer instance with batch_decode support.
 * @param outputTokenIds The full output token IDs from model.generate().
 * @param promptInputIds The prompt token IDs from apply_chat_template().
 * @param skipSpecialTokens Whether to strip special tokens during decoding.
 * @returns The assistant continuation text.
 *
 * @example
 * ```ts
 * const text = decodeAssistantText(tokenizer, outputIds, promptIds, true)
 * ```
 */
function decodeAssistantText(
    tokenizer: { batch_decode: (batch: Tensor, decode_args?: Record<string, unknown>) => string[] },
    outputTokenIds: Tensor,
    promptInputIds: Tensor,
    skipSpecialTokens: boolean,
): string {
    const decodeArgs = skipSpecialTokens ? { skip_special_tokens: true } : {}
    const decodedOutputs = tokenizer.batch_decode(outputTokenIds, decodeArgs)
    const decodedPrompts = tokenizer.batch_decode(promptInputIds, decodeArgs)
    return extractAssistantText(decodedOutputs, decodedPrompts)
}

/**
 * Decodes the prompt input IDs into the exact plain-text sequence sent to generation.
 *
 * @param tokenizer The tokenizer instance with batch_decode support.
 * @param promptInputIds The prompt token IDs from apply_chat_template().
 * @param skipSpecialTokens Whether to strip special tokens during decoding.
 * @returns The decoded prompt text.
 *
 * @example
 * ```ts
 * const prompt = decodePromptText(tokenizer, inputIds, false)
 * ```
 */
function decodePromptText(
    tokenizer: { batch_decode: (batch: Tensor, decode_args?: Record<string, unknown>) => string[] },
    promptInputIds: Tensor,
    skipSpecialTokens: boolean,
): string {
    const decodeArgs = skipSpecialTokens ? { skip_special_tokens: true } : {}
    const decodedPrompts = tokenizer.batch_decode(promptInputIds, decodeArgs)

    if (!isArr(decodedPrompts) || !isStr(decodedPrompts[0])) {
        throw new TypeError('Expected decoded prompt text array from tokenizer.batch_decode().')
    }

    return decodedPrompts[0]
}

export class MicroChat {
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
    async complete(params: MicroChatCompleteParams): Promise<SupportedMessage> {
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

        const promptTextWithSpecialTokens = decodePromptText(
            pipelineInstance.tokenizer,
            inputs.input_ids as Tensor,
            false,
        )

        console.dir({
            promptTextWithSpecialTokens,
            inputs,
        }, { depth: 3 })

        const outputTokenIds = await pipelineInstance.model.generate({
            ...inputs,
            ...streamerConfig,
            ...stoppingCriteriaConfig,
            ...textGenerationConfig,
        })

        // First pass: decode preserving special tokens to detect tool calls
        const rawAssistantText = decodeAssistantText(
            pipelineInstance.tokenizer, outputTokenIds as Tensor, inputs.input_ids as Tensor, false
        )

        try {
            const toolCallsMessage = tryParseToolCalls(rawAssistantText)
            if (toolCallsMessage) {
                return toolCallsMessage
            }
        } catch {
            // Malformed tool call tokens from model output — treat as plain text
        }


        // Second pass: decode stripping special tokens for clean text
        const cleanAssistantText = decodeAssistantText(
            pipelineInstance.tokenizer, outputTokenIds as Tensor, inputs.input_ids as Tensor, true
        )

        return createAssistantMessage(cleanAssistantText)
    }
}