import { ToolCallObj, ToolCallsMessage, ToolResultMessage } from '../Message/types.js'
import { createToolResultMessage } from '../Message/factories.js'
import { FunctionTool } from '../Tools/FunctionTool.js'
import { isToolCallsMessage } from '../Message/guards.js'
import { FunctionToolDeclaration } from './types.js'

export class Tools {
    tools: FunctionTool[] = []

    constructor(...functionToolsArr: FunctionToolDeclaration[]) {
        this.tools = functionToolsArr.map((functionTool) => FunctionTool.from(functionTool))
    }

    static from(functionToolsArr: FunctionToolDeclaration[]): Tools {
        return new Tools(...functionToolsArr)
    }

    addTool(name: string, description: string): FunctionTool {
        // Ensure the tool name is unique
        if (this.tools.some((tool) => tool.name === name)) {
            throw new Error(`A tool with the name "${name}" already exists.`)
        }
        const newTool = new FunctionTool(name, description)
        this.tools.push(newTool)
        return newTool
    }

    async #exeToolCallObj(toolCallObj: ToolCallObj): Promise<ToolResultMessage> {
        const {
            id,
            function: { name, arguments: argsStr },
        } = toolCallObj
        console.debug(`Agent wants to call ${name}(${argsStr})`)
        const tool = this.tools.find((tool) => tool.name === name)
        try {
            if (!tool) {
                throw new Error(`No tool found with the name "${name}"`)
            }
            return createToolResultMessage(
                id,
                await tool.invoke(argsStr),
            )
        } catch (error) {
            return createToolResultMessage(
                id,
                `Error executing tool "${name}": ${error instanceof Error ? error.message : String(error)}`,
            )
        }
    }

    async exeToolCallsMessage(toolsCallMessage: ToolCallsMessage): Promise<ToolResultMessage[]> {
        if (!isToolCallsMessage(toolsCallMessage)) {
            throw new TypeError(
                `Expected a ToolCallsMessage, but got ${JSON.stringify(toolsCallMessage)} (${typeof toolsCallMessage})`,
            )
        }
        const functionCalls = toolsCallMessage.tool_calls.filter((t) => t.type === 'function')
        const toolResultMessages: ToolResultMessage[] = []
        for (const toolCall of functionCalls) {
            toolResultMessages.push(await this.#exeToolCallObj(toolCall))
        }
        return toolResultMessages
    }

    toJSON(): FunctionToolDeclaration[] {
        // Ensure unique tool names
        const toolNames = new Set<string>()
        for (const name of this.tools.map(({ name }) => name)) {
            if (toolNames.has(name)) {
                throw new RangeError(`Duplicate tool names: ${name}`)
            }
            toolNames.add(name)
        }

        return this.tools.map((tool) => tool.toJSON())
    }

    get length(): number {
        return this.tools.length
    }
}
