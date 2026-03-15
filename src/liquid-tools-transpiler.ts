import { isArr, isDef, isPOJO, isStr } from "jty"
import { AssistantMessage, SupportedMessage, ToolCallObj, ToolCallsMessage } from "./Message/types"
import { isAssistantMessage, isMessage, isSupportedMessage, isSupportedMessageArr, isToolCallObj, isToolCallsMessage } from "./Message/guards"
import { Message } from "@huggingface/transformers"
import { createAssistantMessage, createSystemMessage, createToolResultMessage, createUserMessage } from "./Message/factories"

const TOOL_CALL_START_TOKEN = '<|tool_call_start|>'
const TOOL_CALL_END_TOKEN = '<|tool_call_end|>'

/**
 * Removes tool call start and end tokens from a text string.
 * This handles cases where models leak these special tokens into the final generated text.
 *
 * @param text The generated text from the LLM.
 * @returns The text with tool call tokens removed, or the original input if not a string.
 */
export function stripToolCallTokens(text: string): string {
    if (!isStr(text)) return text
    return text.replaceAll(TOOL_CALL_START_TOKEN, '').replaceAll(TOOL_CALL_END_TOKEN, '')
}

export function generateRandomToolCallId(): string {
    return 'call_' + Math.random().toString(36).substring(2, 9)
}

/**
 * Parses Python-style tool call strings (e.g., `[function_name(arg1="value1")]`) into OpenAI-compatible tool call objects.
 * @todo currently this doesn't support multiple function calls
 * @todo currently we don't support values that are objects or arrays
 * @param text The raw generated text that might contain a tool call.
 * @returns An array containing a single tool call object if parsing succeeds, or null if it's not a valid tool call.
 */
export function parsePythonToolCallObj(text: string): ToolCallObj[] {
    if (!isStr(text)) {
        throw new TypeError(`Expected text to be a string, but got ${text} (${typeof text})`)
    }

    text = text.trim()

    // Fallback: If it has special tokens, strip them
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

    const PYTHON_TO_JS: Record<string, boolean | null> = { True: true, False: false, None: null }


    const args: Record<string, unknown> = {}
    if (argsStr.trim().length > 0) {
        // Parse key=value pairs where value can be a quoted string, number, True, False, or None
        const regex = /([a-zA-Z0-9_]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|(True|False|None)|(-?\d+(?:\.\d+)?))/g
        let m
        while ((m = regex.exec(argsStr)) !== null) {
            const key = m[1]
            if (isDef(m[2])) {
                args[key] = m[2]
            } else if (isDef(m[3])) {
                args[key] = m[3]
            } else if (isDef(m[4])) {
                args[key] = PYTHON_TO_JS[m[4]]
            } else if (isDef(m[5])) {
                args[key] = Number(m[5])
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

export function parsePythonToToolCallsMessage(message: AssistantMessage): ToolCallsMessage {
    if (!isAssistantMessage(message)) {
        throw new Error(`Expected an assistant message but got ${JSON.stringify(message)} (${typeof message})`)
    }
    const toolCallObjs = parsePythonToolCallObj(message.content)
    return {
        role: 'assistant',
        tool_calls: toolCallObjs,
    }
}

/**
 * Parses an LFM2 assistant response containing Pythonic tool call tokens into a ToolCallsMessage.
 * Expects the text to contain `<|tool_call_start|>[fn(args)]<|tool_call_end|>` markers.
 * Returns null if the text does not contain a tool call start token.
 *
 * @param text The raw assistant response text.
 * @returns A ToolCallsMessage if a valid tool call is detected, or null otherwise.
 * @throws {SyntaxError} If the start token is found but the end token is missing, or if brackets are missing.
 *
 * @example
 * ```ts
 * tryParseToolCalls('<|tool_call_start|>[get_time()]<|tool_call_end|>') // => ToolCallsMessage
 * tryParseToolCalls('Hello world') // => null
 * ```
 */
export function tryParseToolCalls(text: string): ToolCallsMessage | null {
    if (!isStr(text)) {
        throw new TypeError(`Expected text to be a string, but got ${text} (${typeof text})`)
    }

    const startIdx = text.indexOf(TOOL_CALL_START_TOKEN)
    if (startIdx === -1) return null

    const afterStartIdx = startIdx + TOOL_CALL_START_TOKEN.length
    const endIdx = text.indexOf(TOOL_CALL_END_TOKEN, afterStartIdx)
    if (endIdx === -1) {
        throw new SyntaxError(`Found ${TOOL_CALL_START_TOKEN} but missing ${TOOL_CALL_END_TOKEN} in ${text}`)
    }

    const inner = text.substring(afterStartIdx, endIdx).trim()

    if (!inner.startsWith('[') || !inner.endsWith(']')) {
        throw new SyntaxError(`Expected tool call to be wrapped in brackets, but got: ${inner} in ${text}`)
    }

    const toolCallObjs = parsePythonToolCallObj(inner)

    return { role: 'assistant', tool_calls: toolCallObjs }
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

export function toolCallObjArgumentsToStr(argStr?: string): string {
    if (!argStr) return ''
    const parsedArgs = JSON.parse(argStr)
    return Object.entries(parsedArgs)
        .map(([k, v]) => [k, toolCallObjArgumentValueToPython(v)].join('='))
        .join(', ')
}

export function toolCallsMessageToPython(toolCallsMessage: ToolCallsMessage): AssistantMessage {
    const content = toolCallsMessage.tool_calls
        .filter(isToolCallObj)
        .map((toolCallObj) => {
            return `${toolCallObj.function.name}(${toolCallObjArgumentsToStr(toolCallObj.function.arguments)})`
        })
        .join(',')
    return createAssistantMessage(`${TOOL_CALL_START_TOKEN}[${content}]${TOOL_CALL_END_TOKEN}`)
}

/**
 * Converts one internal SupportedMessage into the plain Message format expected by Liquid models.
 *
 * Why this function exists:
 * The Liquid chat template renders assistant turns from `message.content` only.
 * It can inject tool list tokens (`<|tool_list_start|>...<|tool_list_end|>`) and wrap
 * tool role responses (`<|tool_response_start|>...<|tool_response_end|>`), but it does not
 * convert OpenAI-style `assistant.tool_calls` objects into
 * `<|tool_call_start|>[fn(args)]<|tool_call_end|>`.
 *
 * This adapter is therefore required to preserve stable tool-calling behavior across turns.
 * For assistant tool calls, it serializes `tool_calls` into the canonical Liquid text pattern
 * so the model sees the same format in history that it is expected to produce.
 *
 * When to use:
 * - Before `tokenizer.apply_chat_template(...)` whenever messages may include assistant tool calls.
 * - Keep this conversion even if a chat template is present, unless that template explicitly
 *   consumes `message.tool_calls` and emits tool-call tokens itself.
 *
 * @param message A supported message.
 * @returns A plain Message instance compatible with the Liquid chat template.
 *
 * @example
 * ```ts
 * convertSupportedMessageToLiquidMessage({ role: 'assistant', tool_calls: [toolCall] })
 * // => { role: 'assistant', content: '<|tool_call_start|>[fn()]<|tool_call_end|>' }
 * ```
 */
export function convertSupportedMessageToLiquidMessage(message: SupportedMessage): Message {
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
 * Assistant tool calls are rewritten into the Python-style function call syntax used by LFM2.
 *
 * @param messages An array of supported messages.
 * @returns The formatted messages suitable for the Liquid chat template.
 *
 * @example
 * ```ts
 * convertSupportedMessagesToLiquidMessages([
 *   { role: 'user', content: 'Hi' },
 *   { role: 'assistant', tool_calls: [toolCall] },
 * ])
 * ```
 */
export function convertSupportedMessagesToLiquidMessages(messages: SupportedMessage[]): Message[] {
    if (!isArr(messages)) {
        throw new TypeError(`Expected an array of message objects. Got ${JSON.stringify(messages)}`)
    }

    return messages.map(convertSupportedMessageToLiquidMessage)
}

/**
 * Converts a Liquid model message back into a supported message.
 * Assistant tool call payloads are parsed into ToolCallsMessage objects, while leaked tool-call
 * tokens in normal assistant text are stripped.
 *
 * @param message A plain Message produced by the model.
 * @returns The corresponding supported message.
 *
 * @example
 * ```ts
 * convertLiquidMessageToSupportedMessage({
 *   role: 'assistant',
 *   content: '<|tool_call_start|>[get_time()]<|tool_call_end|>',
 * })
 * ```
 */
export function convertLiquidMessageToSupportedMessage(message: Message): SupportedMessage {
    if (!isMessage(message)) {
        throw new TypeError(`Expected a Message object. Got ${JSON.stringify(message)}`)
    }

    const { role, content } = message

    switch (role) {
        case 'system':
            return createSystemMessage(content)
        case 'user':
            return createUserMessage(content)
        case 'assistant': {
            let toolCallsMessage: ToolCallsMessage | null = null
            const strippedContent = stripToolCallTokens(content).trim()

            try {
                toolCallsMessage = tryParseToolCalls(content)
            } catch {
                toolCallsMessage = null
            }

            if (toolCallsMessage === null && strippedContent.startsWith('[') && strippedContent.endsWith(']')) {
                try {
                    toolCallsMessage = {
                        role: 'assistant',
                        tool_calls: parsePythonToolCallObj(strippedContent),
                    }
                } catch {
                    toolCallsMessage = null
                }
            }

            return toolCallsMessage ?? createAssistantMessage(strippedContent)
        }
        case 'tool':
            if ('tool_call_id' in message && isStr(message.tool_call_id)) {
                return createToolResultMessage(message.tool_call_id, content)
            }
            throw new TypeError(`Expected tool messages to include tool_call_id. Got ${JSON.stringify(message)}`)
        default:
            throw new TypeError(`Unsupported message role: ${JSON.stringify(message)}`)
    }
}

/**
 * Converts an array of Liquid model messages back into supported messages.
 *
 * @param messages A plain Message array.
 * @returns The corresponding supported message array.
 *
 * @example
 * ```ts
 * convertLiquidMessagesToSupportedMessages([{ role: 'user', content: 'Hi' }])
 * ```
 */
export function convertLiquidMessagesToSupportedMessages(messages: Message[]): SupportedMessage[] {
    if (!isArr(messages) || !messages.every(isMessage)) {
        throw new TypeError(`Expected an array of Message objects. Got ${JSON.stringify(messages)}`)
    }

    return messages.map(convertLiquidMessageToSupportedMessage)
}
