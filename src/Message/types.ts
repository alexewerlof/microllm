import { Message } from '@huggingface/transformers'

export type MessageRole = Pick<Message, 'role'>

export interface SystemMessage extends Message {
    role: 'system'
}

export interface UserMessage extends Message {
    role: 'user'
}

export interface AssistantMessage extends Message {
    role: 'assistant'
}

/**
 * Describes a message from an assistant that includes tool call requests.
 */
export interface ToolCallsMessage extends MessageRole {
    role: 'assistant'
    content?: string | null
    tool_calls: ToolCallObj[]
}

/**
 * Describes a request from an LLM to call a specific tool function.
 */
export interface ToolCallObj {
    id: string
    type: 'function'
    function: {
        name: string
        arguments: string
    }
}

export interface ToolResultMessage extends Message {
    role: 'tool'
    tool_call_id: string
}

export type SupportedMessage = SystemMessage | UserMessage | AssistantMessage | ToolCallsMessage | ToolResultMessage

export type SupportedMessageArr = SupportedMessage[]

