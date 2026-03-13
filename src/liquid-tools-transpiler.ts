import { isArr, isDef, isPOJO, isStr } from "jty"
import { AssistantMessage, SupportedMessage, ToolCallObj, ToolCallsMessage } from "./Message/types"
import { isAssistantMessage, isSupportedMessageArr, isToolCallObj, isToolCallsMessage } from "./Message/guards"
import { Message } from "@huggingface/transformers"
import { createAssistantMessage } from "./Message/factories"

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

    const afterStart = startIdx + TOOL_CALL_START_TOKEN.length
    const endIdx = text.indexOf(TOOL_CALL_END_TOKEN, afterStart)
    if (endIdx === -1) {
        throw new SyntaxError(`Found ${TOOL_CALL_START_TOKEN} but missing ${TOOL_CALL_END_TOKEN} in ${text}`)
    }

    const inner = text.substring(afterStart, endIdx).trim()

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
            return `[${toolCallObj.function.name}(${toolCallObjArgumentsToStr(toolCallObj.function.arguments)})]`
        })
        .join('')
    return createAssistantMessage(content)
}

/**
 * Formats OpenAI messages for compatibility with Python-style tool calls.
 * Specifically converts assistant tool_calls into Python-style function call strings.
 *
 * @param messages An array of OpenAI message objects.
 * @returns The formatted messages suitable for the Python-style chat template used by LFM2.
 */
export function convertToolCallsMsgToPython(messages: SupportedMessage[]): Message[] {
    if (!isSupportedMessageArr(messages)) {
        throw new TypeError(`Expected an array of message objects. Got ${JSON.stringify(messages)}`)
    }
    return messages.map((msg) => {
        if (isToolCallsMessage(msg)) {
            return toolCallsMessageToPython(msg)
        }
        return msg
    })
}
