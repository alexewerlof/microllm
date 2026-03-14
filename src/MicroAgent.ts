import { isA, isArr } from 'jty'
import { Tools } from './Tools'
import { MicroLLM } from './MicroLLM'
import { SupportedMessage } from './Message/types'
import { isAssistantMessage, isToolCallsMessage } from './Message/guards'

/**
 * Orchestrates the agentic loop between a MicroLLM and a set of Tools.
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
 * const result = await agent.work(messages, tools)
 * // result is the mutated messages array with tool calls and final answer appended
 * ```
*/
export class MicroAgent {
    /** Maximum number of consecutive tool calls before throwing. */
    static MAX_TOOL_CALLS = 10

    #llm: MicroLLM

    /**
     * @param llm The MicroLLM instance used for generating responses.
     * @throws {TypeError} If llm is not an instance of MicroLLM.
     */
    constructor(llm: MicroLLM) {
        if (!isA(llm, MicroLLM)) {
            throw new TypeError(`Expected llm to be an instance of MicroLLM. Got ${llm} (${typeof llm})`)
        }
        this.#llm = llm
    }

    /**
     * Gets the LLM instance.
     */
    get llm(): MicroLLM {
        return this.#llm
    }

    /**
     * Runs the agentic tool-calling loop.
     *
     * Injects tool declarations as a system message, calls the LLM, and if the response
     * contains a tool call, executes it and feeds the result back. Repeats until
     * the LLM produces a final text answer or MAX_CONSECUTIVE_TOOL_CALLS is reached.
     *
     * All tool call and result messages are returned as a new array.
     *
     * @param messages The conversation messages array (mutated in place).
     * @param tools The tools available for the agent to call.
     * @returns The mutated messages array with all tool calls and the final assistant response appended.
     * @throws {Error} If maximum consecutive tool calls is exceeded.
     *
     * @example
     * ```ts
     * const result = await agent.work(messages, tools)
     * const lastMessage = result[result.length - 1]
     * console.log(lastMessage.content)
     * ```
     */
    async work(messages: SupportedMessage[], tools: Tools): Promise<SupportedMessage[]> {
        if (!isArr(messages)) {
            throw new TypeError(`Expected messages to be an array, but got ${messages} (${typeof messages})`)
        }
        if (!isA(tools, Tools)) {
            throw new TypeError(`Expected tools to be an instance of Tools, but got ${tools} (${typeof tools})`)
        }

        const ret: SupportedMessage[] = []
        for (let iteration = 0; iteration < MicroAgent.MAX_TOOL_CALLS; iteration++) {
            console.log('::::Current messages:', messages)
            const result = await this.#llm.complete({
                messages: [...messages, ...ret],
                tools,
            })

            if (isToolCallsMessage(result)) {
                ret.push(result)
                const toolResults = await tools.exeToolCalls(result)
                ret.push(...toolResults)
                continue
            }

            if (!isAssistantMessage(result)) {
                throw new TypeError(`Expected assistant response from llm.complete(), but got ${JSON.stringify(result)}`)
            }

            ret.push(result)
            return ret
        }

        throw new Error(`Maximum consecutive tool calls exceeded (${MicroAgent.MAX_TOOL_CALLS}).`)
    }
}
