import { isA, isArr, isObj } from 'jty'
import { Tools } from './Tools/index.js'
import { MicroChat } from './MicroChat.js'
import { SupportedMessage } from './Message/types.js'
import { isAssistantMessage, isToolCallsMessage } from './Message/guards.js'

export interface MicroAgentWorkParams {
    /** The conversation messages. */
    messages: SupportedMessage[]
    /** The executable tools available for the agent to call. */
    tools: Tools
    /** Optional abort signal forwarded to the underlying chat completion calls. */
    signal?: AbortSignal
}

/**
 * Orchestrates the agentic loop between a MicroChat and a set of Tools.
 * Sends messages to the LLM, detects tool call requests, executes them,
 * and repeats until the LLM produces a final text answer.
 *
 * @example
 * ```ts
 * const agent = new MicroAgent(llm)
 * const messages: SupportedMessage[] = [
 *   { role: 'system', content: 'You are helpful.' },
 *   { role: 'user', content: 'What time is it?' },
 * ]
 * const tools = new Tools()
 * tools.addTool('get_time', 'Get the current time').func = () => String(new Date())
 * const result = await agent.work({ messages, tools })
 * // result is the mutated messages array with tool calls and final answer appended
 * ```
 */
export class MicroAgent {
    /** Maximum number of consecutive tool calls before throwing. */
    static MAX_TOOL_CALLS = 10

    #microChat: MicroChat

    /**
     * @param microChat The MicroChat instance used for generating responses.
     * @throws {TypeError} If microChat is not an instance of MicroChat.
     */
    constructor(microChat: MicroChat) {
        if (!isA(microChat, MicroChat)) {
            throw new TypeError(
                `Expected microChat to be an instance of MicroChat. Got ${microChat} (${typeof microChat})`,
            )
        }
        this.#microChat = microChat
    }

    /**
     * Gets the MicroChat instance.
     */
    get microChat(): MicroChat {
        return this.#microChat
    }

    /**
     * Runs the agentic tool-calling loop.
     *
     * Injects tool declarations as a system message, calls the MicroChat, and if the response
     * contains a tool call, executes it and feeds the result back. Repeats until
     * the MicroChat produces a final text answer or MAX_CONSECUTIVE_TOOL_CALLS is reached.
     *
     * All tool call and result messages are returned as a new array.
     *
     * @param params The parameters for the work method, including messages, tools, and an optional abort signal.
     * @returns The new messages that are generated as a result of work. The caller can choose to append these to the original messages array or handle them separately.
     * @throws {Error} If maximum consecutive tool calls is exceeded.
     *
     * @example
     * ```ts
     * const result = await agent.work({ messages, tools })
     * const lastMessage = result[result.length - 1]
     * console.log(lastMessage.content)
     * ```
     */
    async work(params: MicroAgentWorkParams): Promise<SupportedMessage[]> {
        if (!isObj(params)) {
            throw new TypeError(`Expected params to be an object, but got ${params} (${typeof params})`)
        }

        const { messages, tools, signal } = params

        if (!isArr(messages)) {
            throw new TypeError(`Expected messages to be an array, but got ${messages} (${typeof messages})`)
        }
        if (!isA(tools, Tools)) {
            throw new TypeError(`Expected tools to be an instance of Tools, but got ${tools} (${typeof tools})`)
        }

        console.debug(JSON.stringify(messages, null, 2))
        const results: SupportedMessage[] = []
        for (let iteration = 0; iteration < MicroAgent.MAX_TOOL_CALLS; iteration++) {
            const result = await this.#microChat.complete({
                messages: [...messages, ...results],
                tools: tools.toJSON(),
                signal,
            })
            console.debug(iteration, JSON.stringify(results, null, 2))

            if (isToolCallsMessage(result)) {
                results.push(result)
                const toolResults = await tools.exeToolCallsMessage(result)
                results.push(...toolResults)
                continue
            }

            if (!isAssistantMessage(result)) {
                throw new TypeError(
                    `Expected assistant response from llm.complete(), but got ${JSON.stringify(result)}`,
                )
            }

            results.push(result)
            return results
        }

        throw new Error(`Maximum consecutive tool calls exceeded (${MicroAgent.MAX_TOOL_CALLS}).`)
    }
}
