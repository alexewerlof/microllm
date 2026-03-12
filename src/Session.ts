import { SupportedMessage } from './Message/types.js'
import { isSupportedMessage } from './Message/guards.js'

/**
 * Represents a conversation history containing messages between the user and the assistant.
 */
export class Session {
    #messages: SupportedMessage[]

    /**
     * Returns a copy of the messages currently stored in the session.
     * External callers should not be able to mutate the internal array.
     */
    get messages(): readonly SupportedMessage[] {
        return [...this.#messages]
    }

    /**
     * Initializes a new Session instance.
     *
     * @param systemPrompt Optional initial system prompt to start the conversation context.
     * @throws {TypeError} If the provided system prompt is not a string.
     */
    constructor() {
        this.#messages = []
    }

    /**
     * Adds a new message wrapper object to the session history.
     *
     * @param message The message wrapper object to add.
     * @throws {TypeError} If the input is not a recognized SupportedMessage object structure.
     */
    addMessage(message: SupportedMessage): void {
        if (!isSupportedMessage(message)) {
            throw new TypeError(`Expected a message, but got ${message} (${typeof message})`)
        }
        this.#messages.push(message)
    }

    /**
     * Converts the internal session message wrappers into a plain array of OpenAI-compatible objects.
     *
     * @returns Array of plain JSON-compatible message objects.
     */
    toJSON(): SupportedMessage[] {
        return JSON.parse(JSON.stringify(this.#messages))
    }
}
