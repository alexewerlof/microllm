import { isFn, isStr } from 'jty'
import { FunctionToolDeclaration, FunctionToolProperties } from './types.js'
import { isFunctionToolDeclaration, isSupportedSimpleType } from './guards.js'

/**
 * A builder class for defining tools that can be executed by LLMs.
 * Fluently define the tool name, description, parameters, and the function mapping.
 */
export class FunctionTool {
    name: string

    /** A description of what the tool does. */
    description: string

    /** The actual JavaScript function to be executed. */
    func?: (...args: unknown[]) => unknown

    /** The value of `this` inside the function when it is invoked */
    thisArg: unknown = undefined

    /** An array of descriptions for each parameter the tool's function accepts. */
    properties: FunctionToolProperties = {}

    /** Indicates if the function parameters allow additional properties not explicitly defined. */
    additionalProperties = false

    /** A flag for strict mode adherence, typically for schema validation by LLMs. */
    strict = false

    static from(declaration: FunctionToolDeclaration): FunctionTool {
        if (!isFunctionToolDeclaration(declaration)) {
            throw new TypeError(`Invalid FunctionToolDeclaration: ${JSON.stringify(declaration)}`)
        }
        const tool = new FunctionTool(declaration.function.name, declaration.function.description)
        if (declaration.function.parameters) {
            for (const [name, prop] of Object.entries(declaration.function.parameters.properties)) {
                const required = declaration.function.parameters.required?.includes(name) ?? false
                if (prop.type === 'array') {
                    tool.properties[name] = {
                        type: 'array',
                        description: prop.description,
                        required,
                        itemsType: prop.itemsType,
                    }
                } else {
                    tool.properties[name] = {
                        type: prop.type,
                        description: prop.description,
                        required,
                    }
                }
            }
            tool.setAdditionalProperties(declaration.function.parameters.additionalProperties ?? false)
        }
        tool.setStrict(declaration.function.strict ?? false)
        return tool
    }

    constructor(name: string, description: string) {
        if (!isStr(name)) {
            throw new TypeError(`Expected tool name to be a string. Got ${name}`)
        }
        if (!isStr(description)) {
            throw new TypeError(`Expected tool description to be a string. Got ${description}`)
        }
        this.name = name
        this.description = description
    }

    /**
     * Attaches the JavaScript function to be executed when the tool is invoked.
     *
     * @param func The target function to call. It usually accepts an object argument.
     * @returns The tool instance for chaining.
     * @throws TypeError If `func` is not a function.
     */
    definition(func: (...args: unknown[]) => unknown): this {
        if (!isFn(func)) {
            throw new TypeError(`Expected tool func to be a function. Got ${func} (${typeof func})`)
        }
        this.func = func
        return this
    }

    /**
     * Sets the `this` context to be used when the tool's function is invoked.
     * @param thisArg The object to use as the `this` context.
     * @returns The tool instance for chaining.
     */
    setThis(thisArg: unknown): this {
        this.thisArg = thisArg
        return this
    }

    /**
     * Reusable builder step to define a tool argument/parameter via the shorthand format.
     *
     * @param name The name of the parameter.
     * @param type The data type of the parameter (e.g., 'string', 'number', 'boolean', 'string[]').
     * @param description An explanation of the parameter's intent.
     * @param required Whether the parameter is required (default: false).
     * @returns The tool instance for chaining.
     */
    param(name: string, type: string, description: string, required = false): this {
        if (!isStr(name)) {
            throw new TypeError(`Expected param name to be a string. Got ${name} (${typeof name})`)
        }
        if (!isStr(type)) {
            throw new TypeError(`Expected param type to be a string. Got ${type} (${typeof type})`)
        }
        if (!isStr(description)) {
            throw new TypeError(`Expected param description to be a string. Got ${description} (${typeof description})`)
        }
        if (isSupportedSimpleType(type)) {
            this.properties[name] = { type, required, description }
        } else {
            if (!type.endsWith('[]')) {
                throw new SyntaxError(`Invalid array type format: ${type}. Expected format like "string[]".`)
            }
            const itemsType = type.slice(0, -2).trim()
            if (!isSupportedSimpleType(itemsType)) {
                throw new SyntaxError(`Unsupported array items type: ${itemsType} in ${type}.`)
            }
            this.properties[name] = { type: 'array', itemsType, required, description }
        }
        return this
    }

    /**
     * Sets whether the underlying function accepts parameters not explicitly defined.
     *
     * @param value True to allow additional properties, false otherwise.
     * @returns The tool instance for chaining.
     */
    setAdditionalProperties(value: boolean): this {
        this.additionalProperties = Boolean(value)
        return this
    }

    /**
     * Sets whether strict mode tracking should be enabled.
     * This is typically used for JSON schema validation enforcements by LLMs.
     *
     * @param value True to enforce strict mode, false otherwise.
     * @returns The tool instance for chaining.
     */
    setStrict(value: boolean): this {
        this.strict = Boolean(value)
        return this
    }

    /**
     * Invokes the tool's function with the given arguments.
     * Arguments are expected to be a JSON string representing an object.
     *
     * @param argsStr - A JSON string representing the arguments object for the function.
     * @returns A promise that resolves to a string representation of the function's result.
     */
    async invoke(argsStr: string): Promise<string> {
        try {
            if (!isFn(this.func)) {
                throw new TypeError(`Tool ${this.name} hasn't defined a function body.`)
            }
            const args = JSON.parse(argsStr)
            const result = await this.func.call(this.thisArg, args)
            console.debug(
                `Successfully executed ${this.name}(${argsStr}) => ${typeof result}\n${JSON.stringify(result, null, 2)}`,
            )
            return JSON.stringify(result)
        } catch (error) {
            return `Error executing ${this.name}(${argsStr}): ${error instanceof Error ? error.message : String(error)}`
        }
    }

    /**
     * Gets the tool's description in a format suitable for LLM function calling.
     * @returns The structured description of the tool.
     */
    toJSON(): FunctionToolDeclaration {
        const required = Object.entries(this.properties)
            .filter(([, p]) => p.required)
            .map(([name]) => name)

        return {
            type: 'function',
            function: {
                name: this.name,
                description: this.description,
                parameters: {
                    type: 'object',
                    properties: this.properties,
                    required,
                    additionalProperties: this.additionalProperties,
                },
                strict: this.strict,
            },
        }
    }
}
