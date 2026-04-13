import { Message } from '@huggingface/transformers'
import { isArr, isDef, isObj, isStr } from 'jty'
import { createAssistantMessage } from '../Message/factories.js'
import { SupportedMessage, ToolCallObj, ToolCallsMessage, ToolResultMessage } from '../Message/types.js'
import { BeforeChatTemplateParams, BeforeChatTemplateResult, AfterDecodeParams, ChatModelAdapter } from './types.js'

const GEMMA4_TOOL_CALL_START_TOKEN = '<|tool_call>'
// If needed in the future: const GEMMA4_TOOL_CALL_END_TOKEN = '<tool_call|>'
const GEMMA4_QUOTE_TOKEN = '<|"|>'
const GEMMA4_FALLBACK_CHAT_TEMPLATE =
    "{{- bos_token -}}{%- for message in messages -%}{%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}{{- '<|turn>' + role + '\\n' -}}{%- if message['content'] is string -%}{{- message['content'] | trim -}}{%- endif -%}{{- '<turn|>\\n' -}}{%- endfor -%}{%- if add_generation_prompt -%}{{- '<|turn>model\\n' -}}{%- endif -%}"

interface Gemma4ToolResponse {
    name: string
    response: unknown
}

interface Gemma4AssistantWithTools extends Message {
    role: 'assistant'
    tool_calls?: Array<{
        function: {
            name: string
            arguments: Record<string, unknown>
        }
    }>
    tool_responses?: Gemma4ToolResponse[]
}

/**
 * Removes Gemma 4 thought-channel blocks from decoded text.
 *
 * @example
 * ```ts
 * const text = stripGemma4Thinking('A<|channel>thought\nB<channel|>C')
 * ```
 */
function stripGemma4Thinking(text: string): string {
    if (!isStr(text)) {
        throw new TypeError(`Expected text to be a string, but got ${text} (${typeof text})`)
    }

    return text.replace(/<\|channel>thought\s*[\s\S]*?<channel\|>/g, '').trim()
}

function parsePrimitiveValue(rawValue: string): unknown {
    const trimmed = rawValue.trim()

    if (trimmed.startsWith('"') && trimmed.endsWith('"')) {
        return trimmed.slice(1, -1)
    }

    if (trimmed === 'true' || trimmed === 'True') return true
    if (trimmed === 'false' || trimmed === 'False') return false
    if (trimmed === 'null' || trimmed === 'None') return null

    const parsedNum = Number(trimmed)
    if (Number.isFinite(parsedNum) && trimmed !== '') {
        return parsedNum
    }

    return trimmed
}

function splitTopLevelCommaArgs(text: string): string[] {
    if (!isStr(text)) {
        throw new TypeError(`Expected text to be a string, but got ${text} (${typeof text})`)
    }

    const pairs: string[] = []
    let current = ''
    let quote: '"' | "'" | null = null

    for (let i = 0; i < text.length; i++) {
        const char = text[i]

        if ((char === '"' || char === "'") && (i === 0 || text[i - 1] !== '\\')) {
            quote = quote === char ? null : quote === null ? (char as '"' | "'") : quote
            current += char
            continue
        }

        if (char === ',' && quote === null) {
            if (current.trim().length > 0) {
                pairs.push(current.trim())
            }
            current = ''
            continue
        }

        current += char
    }

    if (current.trim().length > 0) {
        pairs.push(current.trim())
    }

    return pairs
}

/**
 * Parses Gemma 4 tool-call arguments from `key:value` syntax into an object.
 *
 * @example
 * ```ts
 * const args = parseGemma4Arguments('location:<|"|>Seoul<|"|>,unit:<|"|>celsius<|"|>')
 * ```
 */
function parseGemma4Arguments(rawArgs: string): Record<string, unknown> {
    if (!isStr(rawArgs)) {
        throw new TypeError(`Expected rawArgs to be a string, but got ${rawArgs} (${typeof rawArgs})`)
    }

    const normalized = rawArgs.replaceAll(GEMMA4_QUOTE_TOKEN, '"').trim()
    if (normalized.length === 0) {
        return {}
    }

    return splitTopLevelCommaArgs(normalized).reduce<Record<string, unknown>>((acc, pair) => {
        const separatorIdx = pair.indexOf(':')
        if (separatorIdx === -1) {
            throw new SyntaxError(`Expected key:value pair in Gemma 4 arguments, but got ${pair}`)
        }

        const key = pair.slice(0, separatorIdx).trim()
        const value = pair.slice(separatorIdx + 1)

        if (key.length === 0) {
            throw new SyntaxError(`Expected non-empty argument key in pair: ${pair}`)
        }

        acc[key] = parsePrimitiveValue(value)
        return acc
    }, {})
}

/**
 * Parses Gemma 4 tool-call tokens into canonical ToolCallsMessage objects.
 *
 * @example
 * ```ts
 * const msg = tryParseGemma4ToolCallsMessage('<|tool_call>call:get_time{}<tool_call|>')
 * ```
 */
function tryParseGemma4ToolCallsMessage(rawAssistantResponse: string): ToolCallsMessage {
    if (!isStr(rawAssistantResponse)) {
        throw new TypeError(
            `Expected assistant response to be a string, but got ${rawAssistantResponse} (${typeof rawAssistantResponse})`,
        )
    }

    if (!rawAssistantResponse.includes(GEMMA4_TOOL_CALL_START_TOKEN)) {
        throw new SyntaxError(
            `Expected ${GEMMA4_TOOL_CALL_START_TOKEN} token in response, but got: ${rawAssistantResponse}`,
        )
    }

    const matches = Array.from(
        rawAssistantResponse.matchAll(/<\|tool_call>call:([a-zA-Z0-9_]+)\{([\s\S]*?)\}<tool_call\|>/g),
    )

    if (!isArr(matches) || matches.length === 0) {
        throw new SyntaxError(`Could not parse Gemma 4 tool calls from response: ${rawAssistantResponse}`)
    }

    const toolCalls: ToolCallObj[] = matches.map((match, index) => {
        const functionName = match[1]
        const rawArgs = match[2]

        return {
            id: `call_gemma4_${index}_${Math.random().toString(36).substring(2, 8)}`,
            type: 'function',
            function: {
                name: functionName,
                arguments: JSON.stringify(parseGemma4Arguments(rawArgs)),
            },
        }
    })

    return {
        role: 'assistant',
        content: null,
        tool_calls: toolCalls,
    }
}

function parseJsonIfPossible(value: string): unknown {
    try {
        return JSON.parse(value)
    } catch {
        return value
    }
}

function toolCallObjToGemma4ToolCallObj(toolCall: ToolCallObj): { function: { name: string; arguments: Record<string, unknown> } } {
    return {
        function: {
            name: toolCall.function.name,
            arguments: parseJsonIfPossible(toolCall.function.arguments) as Record<string, unknown>,
        },
    }
}

function toGemma4AssistantToolResponse(message: ToolResultMessage, toolCallName: string): Gemma4AssistantWithTools {
    return {
        role: 'assistant',
        content: '',
        tool_responses: [
            {
                name: toolCallName,
                response: parseJsonIfPossible(message.content as string),
            },
        ],
    }
}

/**
 * Converts canonical messages into Gemma 4-compatible chat-template messages.
 *
 * @example
 * ```ts
 * const prepared = convertSupportedMessagesToGemma4Messages(messages)
 * ```
 */
function convertSupportedMessagesToGemma4Messages(messages: SupportedMessage[]): Message[] {
    if (!isArr(messages)) {
        throw new TypeError(`Expected messages to be an array, but got ${messages} (${typeof messages})`)
    }

    const toolCallNameById = new Map<string, string>()

    return messages.map((message) => {
        if (!isObj(message) || !isStr(message.role)) {
            throw new TypeError(`Expected a message object with a role, but got ${JSON.stringify(message)}`)
        }

        if (message.role === 'assistant' && isDef((message as ToolCallsMessage).tool_calls)) {
            const toolCallsMessage = message as ToolCallsMessage
            const gemmaToolCalls = toolCallsMessage.tool_calls.map((toolCall) => {
                toolCallNameById.set(toolCall.id, toolCall.function.name)
                return toolCallObjToGemma4ToolCallObj(toolCall)
            })

            return {
                role: 'assistant',
                content: toolCallsMessage.content ?? '',
                tool_calls: gemmaToolCalls,
            } as Gemma4AssistantWithTools as Message
        }

        if (message.role === 'tool') {
            const toolResultMessage = message as ToolResultMessage
            const toolCallName = toolCallNameById.get(toolResultMessage.tool_call_id)
            if (!isStr(toolCallName) || toolCallName.length === 0) {
                throw new TypeError(
                    `Could not map tool_call_id ${toolResultMessage.tool_call_id} to a known tool call name for Gemma 4 tool_responses.`,
                )
            }

            return toGemma4AssistantToolResponse(toolResultMessage, toolCallName) as Message
        }

        return {
            role: message.role,
            content: message.content,
        } as Message
    })
}

/**
 * Adapter for Gemma 4 tool-calling and chat semantics.
 *
 * @example
 * ```ts
 * const chat = new MicroChat(factory, new Gemma4Adapter())
 * ```
 */
export class Gemma4Adapter implements ChatModelAdapter {
    name = 'gemma4'

    onBeforeChatTemplate(params: BeforeChatTemplateParams): BeforeChatTemplateResult {
        return {
            messages: convertSupportedMessagesToGemma4Messages(params.messages),
            templateOptions: {
                chat_template: GEMMA4_FALLBACK_CHAT_TEMPLATE,
                tools: params.tools,
            },
        }
    }

    onAfterDecode(params: AfterDecodeParams): SupportedMessage {
        try {
            return tryParseGemma4ToolCallsMessage(params.rawAssistantContent)
        } catch {
            return createAssistantMessage(stripGemma4Thinking(params.cleanAssistantContent))
        }
    }
}

export const _test = {
    stripGemma4Thinking,
    parseGemma4Arguments,
    tryParseGemma4ToolCallsMessage,
    convertSupportedMessagesToGemma4Messages,
    GEMMA4_FALLBACK_CHAT_TEMPLATE,
}
