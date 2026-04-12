import { Message, StoppingCriteriaList, Tensor, TextGenerationPipeline, TextStreamer } from '@huggingface/transformers'

type TextGenerationOptions = NonNullable<Parameters<TextGenerationPipeline['_call']>[1]>
import { PipelineFactory } from './PipelineFactory.js'
import { isArr, isDef, isObj, isPOJO, isStr } from 'jty'
import { normalizeMessageArray } from './utilities/normalization.js'
import { SignalStoppingCriteria } from './utilities/SignalStoppingCriteria.js'
import { FunctionToolDeclaration, isFunctionToolDeclaration } from './Tools/index.js'
import { SupportedMessage } from './Message/types.js'
import { createAssistantMessage } from './Message/factories.js'
import { convertSupportedMessagesToLiquidMessages, tryParseAsToolCallsMessage } from './liquid-tools-transpiler.js'

const defaultTextGenerationConfig: Partial<TextGenerationOptions> = {
    max_new_tokens: 512,
    temperature: 0.5,
    top_p: 0.5,
}

export interface MicroChatCompleteParams {
    /** The conversation messages. */
    messages: SupportedMessage[]
    /** Optional text generation config like temperature, top_p, etc. */
    config?: Partial<TextGenerationOptions>
    /** Optional abort signal. */
    signal?: AbortSignal
    /** Optional tool declarations whose schemas are injected into the prompt. */
    tools?: FunctionToolDeclaration[]
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
 * It is primarily used for debugging to see what we're sending to the model.
 *
 * @param tokenizer The tokenizer instance with batch_decode support.
 * @param promptInputIds The prompt token IDs from apply_chat_template().
 * @param skipSpecialTokens Whether to strip special tokens during decoding.
 * @returns The decoded prompt text.
 *
 * @example
 * ```ts
 * _printDecodePromptText(tokenizer, inputIds, false)
 * ```
 */
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function _printDecodePromptText(
    tokenizer: { batch_decode: (batch: Tensor, decode_args?: Record<string, unknown>) => string[] },
    promptInputIds: Tensor,
    skipSpecialTokens: boolean,
): void {
    const decodeArgs = skipSpecialTokens ? { skip_special_tokens: true } : {}
    const decodedPrompts = tokenizer.batch_decode(promptInputIds, decodeArgs)

    if (!isArr(decodedPrompts) || !isStr(decodedPrompts[0])) {
        throw new TypeError('Expected decoded prompt text array from tokenizer.batch_decode().')
    }

    const promptTextWithSpecialTokens = decodedPrompts[0]

    console.debug(promptTextWithSpecialTokens)
}

export class MicroChat {
    pipelineFactory: PipelineFactory<'text-generation'>

    /**
     * Creates a chat instance from a pre-configured text generation pipeline factory.
     * The caller owns model lifecycle concerns such as eager loading and unloading.
     *
     * @param pipelineFactory Factory that resolves the text generation pipeline.
     *
     * @example
     * ```ts
     * const factory = new PipelineFactory('text-generation', 'onnx-community/LFM2-1.2B-Tool-ONNX', { dtype: 'q4' })
     * const llm = new MicroChat(factory)
     * ```
     */
    constructor(pipelineFactory: PipelineFactory<'text-generation'>) {
        this.pipelineFactory = pipelineFactory
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
     * const tools: FunctionToolDeclaration[] = [{
     *   type: 'function',
     *   function: {
     *     name: 'get_time',
     *     description: 'Get the current time',
     *     parameters: {
     *       type: 'object',
     *       properties: {},
     *       required: [],
     *       additionalProperties: false,
     *     },
     *   },
     * }]
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
            if (!Array.isArray(tools) || !tools.every(isFunctionToolDeclaration)) {
                throw new TypeError(
                    `Expected tools to be an array of FunctionToolDeclaration objects, but got ${JSON.stringify(tools)} (${typeof tools})`,
                )
            }
        }

        if (isDef(config)) {
            if (!isPOJO(config)) {
                throw new TypeError(
                    `When specified, completion config should be an object. Got ${config} (${typeof config})`,
                )
            }
        }

        const textGenerationConfig: Partial<TextGenerationOptions> = {
            ...defaultTextGenerationConfig,
            ...config,
        }

        const stoppingCriteriaConfig: {
            stopping_criteria?: StoppingCriteriaList
        } = {}
        if (isDef(signal)) {
            if (signal.aborted) {
                throw new Error('The signal is already aborted. Cannot start generation.')
            }
            stoppingCriteriaConfig.stopping_criteria = SignalStoppingCriteria.createStoppingCriteriaList(signal)
        }

        const pipelineInstance = await this.pipelineFactory.getPipeline()

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
            tools,
            add_generation_prompt: true,
            return_dict: true,
        }) as Record<string, unknown>

        // _printDecodePromptText(pipelineInstance.tokenizer, inputs.input_ids as Tensor, false)

        const outputTokenIds = await pipelineInstance.model.generate({
            ...inputs,
            ...streamerConfig,
            ...stoppingCriteriaConfig,
            ...textGenerationConfig,
        })

        // First pass: decode preserving special tokens to detect tool calls
        const rawAssistantContent = decodeAssistantText(
            pipelineInstance.tokenizer,
            outputTokenIds as Tensor,
            inputs.input_ids as Tensor,
            false,
        )

        try {
            return tryParseAsToolCallsMessage(rawAssistantContent)
        } catch {
            // If parsing fails, we fall back to regular assistant message
            const cleanAssistantText = decodeAssistantText(
                pipelineInstance.tokenizer,
                outputTokenIds as Tensor,
                inputs.input_ids as Tensor,
                true,
            )

            return createAssistantMessage(cleanAssistantText)
        }
    }
}
