import { Message } from '@huggingface/transformers'
import { isArr } from 'jty'

/**
 * Moves all system messages to the start of the message array, while preserving the order of the other messages.
 * @param messageArr The array of messages to process.
 * @returns A new array with system messages at the start.
 */
function moveAllSystemMessagesToStart(messageArr: Message[]): Message[] {
    if (!isArr(messageArr)) {
        throw new TypeError(`Expected an array. Got ${messageArr} (${typeof messageArr})`)
    }
    const systemMessages: Message[] = []
    const theRest: Message[] = []
    for (const message of messageArr) {
        if (message.role === 'system') {
            systemMessages.push(message)
        } else {
            theRest.push(message)
        }
    }
    return [...systemMessages, ...theRest]
}

/**
 * Merges consecutive messages from the same role into a single message with concatenated content.
 * @param messageArr The array of messages to process.
 * @returns A new array with consecutive messages from the same role merged.
 */
function mergeConsecutiveMessages(messageArr: Message[]): Message[] {
    if (!isArr(messageArr)) {
        throw new TypeError(`Expected an array. Got ${messageArr} (${typeof messageArr})`)
    }

    const targetRoles = ['user', 'system']
    const ret: Message[] = []
    for (const message of messageArr) {
        const lastMessage = ret[ret.length - 1]
        const { role } = message
        if (targetRoles.includes(role) && role === lastMessage?.role) {
            ret.pop()
            ret.push({
                role,
                content: [lastMessage.content, message.content].join('\n'),
            })
        } else {
            ret.push(message)
        }
    }
    return ret
}

/**
 * Goes through the messages and ensure that:
 * 1. All the system messages are at the beginning of the array
 * 2. The messages from the same role have their content concatenated together
 * 3. The order of the messages is preserved as much as possible
 * @param messageArray The array of messages to process.
 * @returns A new array with normalized messages.
 */
export function normalizeMessageArray(messageArray: Message[]): Message[] {
    return mergeConsecutiveMessages(moveAllSystemMessagesToStart(messageArray))
}

export const _test = { moveAllSystemMessagesToStart, mergeMessages: mergeConsecutiveMessages }