import { Message } from '@huggingface/transformers'
import { isArr, isDef, isPOJO, isStr } from 'jty'
import { createAssistantMessage } from '../Message/factories.js'
import { isSupportedMessage, isToolCallObj, isToolCallsMessage } from '../Message/guards.js'
import { AssistantMessage, SupportedMessage, ToolCallObj, ToolCallsMessage } from '../Message/types.js'
import { AfterDecodeParams, BeforeChatTemplateParams, BeforeChatTemplateResult, ChatModelAdapter } from './types.js'

const TOOL_CALL_START_TOKEN = '<|tool_call_start|>'
const TOOL_CALL_END_TOKEN = '<|tool_call_end|>'

/**
 * Removes tool call start and end tokens from a text string.
 *
 * @param text The generated text from the LLM.
 * @returns The text with tool call tokens removed.
 *
 * @example
 * ```ts
 * stripToolCallTokens('Hello <|tool_call_start|>world<|tool_call_end|>!')
 * ```
 */
function stripToolCallTokens(text: string): string {
    if (!isStr(text)) return text
    return text.replaceAll(TOOL_CALL_START_TOKEN, '').replaceAll(TOOL_CALL_END_TOKEN, '')
}

function generateRandomToolCallId(): string {
    return 'call_' + Math.random().toString(36).substring(2, 9)
}

/**
 * Parses Python-style tool call strings into OpenAI-compatible tool call objects.
 *
 * @param text The raw generated text that might contain a tool call.
 * @returns Parsed tool calls.
 *
 * @example
 * ```ts
 * parsePythonToolCallObj('[get_time()]')
 * ```
 */
function parsePythonToolCallObj(text: string): ToolCallObj[] {
    if (!isStr(text)) {
        throw new TypeError(`Expected text to be a string, but got ${text} (${typeof text})`)
    }

    text = text.trim()
    text = stripToolCallTokens(text).trim()

    if (!text.startsWith('[') || !text.endsWith(']')) {
        throw new SyntaxError(`Expected a tool call block, but got ${text}`)
    }

    const body = text.slice(1, -1).trim()
    const match = body.match(/^([a-zA-Z0-9_]+)\((.*)\)$/)
    if (!match) {
        throw new SyntaxError(`Could not find function name: ${text}`)
    }

    const [, functionName, argsStr] = match
    const pythonToJs: Record<string, boolean | null> = { True: true, False: false, None: null }
    const args: Record<string, unknown> = {}

    if (argsStr.trim().length > 0) {
        const regex = /([a-zA-Z0-9_]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|(True|False|None)|(-?\d+(?:\.\d+)?))/g
        let parsedMatch
        while ((parsedMatch = regex.exec(argsStr)) !== null) {
            const key = parsedMatch[1]
            if (isDef(parsedMatch[2])) {
                args[key] = parsedMatch[2]
            } else if (isDef(parsedMatch[3])) {
                args[key] = parsedMatch[3]
            } else if (isDef(parsedMatch[4])) {
                args[key] = pythonToJs[parsedMatch[4]]
            } else if (isDef(parsedMatch[5])) {
                args[key] = Number(parsedMatch[5])
            }
        }
    }

    return [
        {
            id: generateRandomToolCallId(),
            type: 'function',
            function: {
                name: functionName,
                arguments: JSON.stringify(args),
            },
        },
    ]
}

function indexOfToolCallStartToken(rawAssistantResponse: string): number {
    if (!isStr(rawAssistantResponse)) {
        throw new TypeError(
            `Expected assistant response to be a string, but got ${rawAssistantResponse} (${typeof rawAssistantResponse})`,
        )
    }
    return rawAssistantResponse.indexOf(TOOL_CALL_START_TOKEN)
}

/**
 * Parses an LFM2 assistant response containing Pythonic tool call tokens.
 *
 * @param rawAssistantResponse The raw assistant response text.
 * @returns A ToolCallsMessage if a valid tool call is detected.
 *
 * @example
 * ```ts
 * tryParseAsToolCallsMessage('<|tool_call_start|>[get_time()]<|tool_call_end|>')
 * ```
 */
function tryParseAsToolCallsMessage(rawAssistantResponse: string): ToolCallsMessage {
    const startIdx = indexOfToolCallStartToken(rawAssistantResponse)
    if (startIdx === -1) {
        throw new SyntaxError(
            `Expected a tool call start token in the assistant response, but got: ${rawAssistantResponse}`,
        )
    }

    const afterStartIdx = startIdx + TOOL_CALL_START_TOKEN.length
    const endIdx = rawAssistantResponse.indexOf(TOOL_CALL_END_TOKEN, afterStartIdx)
    if (endIdx === -1) {
        throw new SyntaxError(
            `Found ${TOOL_CALL_START_TOKEN} but missing ${TOOL_CALL_END_TOKEN} in ${rawAssistantResponse}`,
        )
    }

    const inner = rawAssistantResponse.substring(afterStartIdx, endIdx).trim()
    if (!inner.startsWith('[') || !inner.endsWith(']')) {
        throw new SyntaxError(`Expected tool call to be wrapped in brackets, but got: ${inner} in ${rawAssistantResponse}`)
    }

    return { role: 'assistant', tool_calls: parsePythonToolCallObj(inner) }
}

function toolCallObjArgumentValueToPython(value: unknown): string {
    switch (typeof value) {
        case 'string':
            return `"${value.replace(/"/g, '\\"')}"`
        case 'number':
            return String(value)
        case 'boolean':
            return value ? 'True' : 'False'
        case 'object':
            if (value === null) {
                return 'None'
            }
            if (isArr(value) || isPOJO(value)) {
                return JSON.stringify(value)
            }
            throw new TypeError(`Unsupported object type: ${value}`)
        default:
            throw new TypeError(`Unsupported argument type: ${typeof value}`)
    }
}

function toolCallObjArgumentsToStr(argStr?: string): string {
    if (!argStr) return ''
    const parsedArgs = JSON.parse(argStr)
    return Object.entries(parsedArgs)
        .map(([k, v]) => [k, toolCallObjArgumentValueToPython(v)].join('='))
        .join(', ')
}

function toolCallsMessageToPython(toolCallsMessage: ToolCallsMessage): AssistantMessage {
    const content = toolCallsMessage.tool_calls
        .filter(isToolCallObj)
        .map(({ function: { name, arguments: argStr } }) => `${name}(${toolCallObjArgumentsToStr(argStr)})`)
        .join(',')

    return createAssistantMessage(`${TOOL_CALL_START_TOKEN}[${content}]${TOOL_CALL_END_TOKEN}`)
}

function convertSupportedMessageToLiquidMessage(message: SupportedMessage): Message {
    if (!isSupportedMessage(message)) {
        throw new TypeError(`Expected a supported message object. Got ${JSON.stringify(message)}`)
    }

    if (isToolCallsMessage(message)) {
        return toolCallsMessageToPython(message)
    }

    return {
        role: message.role,
        content: message.content,
    }
}

/**
 * Converts supported messages into the plain Message format expected by Liquid models.
 *
 * @param messages An array of supported messages.
 * @returns The formatted messages suitable for the Liquid chat template.
 *
 * @example
 * ```ts
 * convertSupportedMessagesToLiquidMessages([{ role: 'user', content: 'Hi' }])
 * ```
 */
function convertSupportedMessagesToLiquidMessages(messages: SupportedMessage[]): Message[] {
    if (!isArr(messages)) {
        throw new TypeError(`Expected an array of message objects. Got ${JSON.stringify(messages)}`)
    }

    return messages.map(convertSupportedMessageToLiquidMessage)
}

/**
 * Adapter for Liquid tool-calling semantics.
 *
 * @example
 * ```ts
 * const chat = new MicroChat(factory, new LiquidAdapter())
 * ```
 */
export class LiquidAdapter implements ChatModelAdapter {
    name = 'liquid'

    onBeforeChatTemplate(params: BeforeChatTemplateParams): BeforeChatTemplateResult {
        return {
            messages: convertSupportedMessagesToLiquidMessages(params.messages),
            templateOptions: {
                tools: params.tools,
            },
        }
    }

    onAfterDecode(params: AfterDecodeParams) {
        try {
            return tryParseAsToolCallsMessage(params.rawAssistantContent)
        } catch {
            return createAssistantMessage(params.cleanAssistantContent)
        }
    }
}

export const _test = {
    stripToolCallTokens,
    parsePythonToolCallObj,
    tryParseAsToolCallsMessage,
    convertSupportedMessagesToLiquidMessages,
}
