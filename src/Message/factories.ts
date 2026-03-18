import { isToolCallObj } from './guards.js'
import {
    AssistantMessage,
    SystemMessage,
    ToolCallObj,
    ToolCallsMessage,
    ToolResultMessage,
    UserMessage,
} from './types.js'

export function createSystemMessage(content: string): SystemMessage {
    return {
        role: 'system',
        content,
    }
}

export function createUserMessage(content: string): UserMessage {
    return {
        role: 'user',
        content,
    }
}

export function createAssistantMessage(content: string): AssistantMessage {
    return {
        role: 'assistant',
        content,
    }
}

export function createToolCallsMessage(tool_calls: ToolCallObj[]): ToolCallsMessage {
    for (const toolCall of tool_calls) {
        if (!isToolCallObj(toolCall)) {
            throw new TypeError(`Invalid tool call object: ${JSON.stringify(toolCall)}`)
        }
    }

    return {
        role: 'assistant',
        content: null,
        tool_calls,
    }
}

/**
 * Creates a ToolResultMessage object.
 * @param toolCallId - The ID of the tool call.
 * @param content - The result content from the tool execution.
 * @returns The constructed ToolResultMessage.
 */
export function createToolResultMessage(toolCallId: string, content: string): ToolResultMessage {
    return {
        role: 'tool',
        tool_call_id: toolCallId,
        content,
    }
}
