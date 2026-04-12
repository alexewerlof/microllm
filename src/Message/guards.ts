import { hasProp, inArr, isArr, isArrLen, isPOJO, isStr } from 'jty'
import {
    AssistantMessage,
    MessageRole,
    SupportedMessage,
    SupportedMessageArr,
    SystemMessage,
    ToolCallObj,
    ToolCallsMessage,
    ToolResultMessage,
    UserMessage,
} from './types.js'
import { Message } from '@huggingface/transformers'

export const SUPPORTED_ROLES = ['system', 'user', 'assistant', 'tool'] as const

/**
 * Checks whether an object is a structurally valid BaseWMsg format.
 * Optionally enforces a specific `role` validation.
 *
 * @param x The object to evaluate.
 * @returns True if the object matches the criteria, false otherwise.
 */
export function isMessageRole(x: unknown): x is MessageRole {
    if (!hasProp(x, 'role') || !isStr(x.role)) {
        return false
    }
    return inArr(x.role, SUPPORTED_ROLES)
}

/**
 * Verifies if an object is a valid Message object.
 * Validates both the base message structure and ensures it contains a string payload.
 *
 * @param x The object to validate.
 * @returns True if it is a structurally sound Message, false otherwise.
 */
export function isMessage(x: unknown): x is Message {
    if (!isMessageRole(x)) {
        return false
    }
    return hasProp(x, 'content') && (isStr(x.content) || isArr(x.content))
}

/**
 * Checks if a given object represents a valid SystemMessage structure.
 *
 * @param x The object to validate.
 * @returns True if it is a valid SystemMessage representation, false otherwise.
 */
export function isSystemMessage(x: unknown): x is SystemMessage {
    return isMessage(x) && x.role === 'system'
}

/**
 * Checks if a given object represents a valid UserMessage structure.
 *
 * @param x The object to validate.
 * @returns True if it is a valid UserMessage representation, false otherwise.
 */
export function isUserMessage(x: unknown): x is UserMessage {
    return isMessage(x) && x.role === 'user'
}

/**
 * Checks if a given object represents a valid AssistantMessage structure.
 *
 * @param x The object to validate.
 * @returns True if it is a valid AssistantMessage representation, false otherwise.
 */
export function isAssistantMessage(x: unknown): x is AssistantMessage {
    return isMessage(x) && x.role === 'assistant'
}

/**
 * Verifies if an object is a structurally valid ToolCallsMessage representation.
 *
 * @param x The object to validate.
 * @returns True if the object matches the schema, false otherwise.
 */
export function isToolCallsMessage(x: unknown): x is ToolCallsMessage {
    if (!isMessageRole(x) || x.role !== 'assistant') {
        return false
    }
    return isArrLen((x as ToolCallsMessage).tool_calls, 1) && (x as ToolCallsMessage).tool_calls.every(isToolCallObj)
}

/**
 * Validates an individual ToolCallObj against the required structure.
 *
 * @param x A single tool call object.
 * @returns True if valid structurally, false otherwise.
 */
export function isToolCallObj(x: unknown): x is ToolCallObj {
    if (!isPOJO(x)) {
        return false
    }
    return (
        hasProp(x, 'id', 'type', 'function') &&
        isStr(x.id) &&
        x.type === 'function' &&
        hasProp(x.function, 'name', 'arguments') &&
        isStr(x.function.name) &&
        isStr(x.function.arguments)
    )
}

/**
 * Checks if a given object represents a valid ToolResultMessage structure.
 *
 * @param x The object to validate.
 * @returns True if it is a valid ToolResultMessage representation, false otherwise.
 */
export function isToolResultMessage(x: unknown): x is ToolResultMessage {
    if (!isMessage(x) || x.role !== 'tool') {
        return false
    }
    return hasProp(x, 'tool_call_id') && isStr(x.tool_call_id)
}

export function isSupportedMessage(x: unknown): x is SupportedMessage {
    return (
        isSystemMessage(x) ||
        isUserMessage(x) ||
        isAssistantMessage(x) ||
        isToolCallsMessage(x) ||
        isToolResultMessage(x)
    )
}

export function isSupportedMessageArr(x: unknown): x is SupportedMessageArr {
    return isArr(x) && x.every(isSupportedMessage)
}
