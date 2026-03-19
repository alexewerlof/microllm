import { isBool, isDef, isFn, isObj, isStr } from 'jty'

export interface FunctionToolBaseProperty {
    /** The name of the parameter. */
    name: string
    /** The data type of the parameter (e.g., 'string', 'number', 'boolean'). */
    type: string
    /** Whether this parameter is required. */
    required?: boolean
    /** A human-readable description of the parameter. */
    description?: string
}

export function isFunctionToolBaseProperty(obj: unknown): obj is FunctionToolBaseProperty {
    if (!isObj(obj)) {
        return false
    }

    const { name, type, required, description } = obj as Partial<FunctionToolBaseProperty>

    if (!isStr(name)) {
        return false
    }

    if (!isStr(type)) {
        return false
    }

    if (isDef(required) && !isBool(required)) {
        return false
    }

    if (isDef(description) && !isStr(description)) {
        return false
    }

    return true
}

export interface FunctionToolSimpleProperty extends FunctionToolBaseProperty {
    name: 'string' | 'number' | 'boolean'
}

export function isFunctionToolSimpleProperty(obj: unknown): obj is FunctionToolSimpleProperty {
    if (!isFunctionToolBaseProperty(obj)) {
        return false
    }

    if (!['string', 'number', 'boolean'].includes(obj.type)) {
        return false
    }

    return true
}

export interface FunctionToolArrayProperty extends FunctionToolBaseProperty {
    type: 'array'
    /** When `type` is 'array', the items type (e.g., 'string'). */
    itemsType: string
}

export function isFunctionToolArrayProperty(obj: unknown): obj is FunctionToolArrayProperty {
    if (!isFunctionToolBaseProperty(obj)) {
        return false
    }

    const { type, itemsType } = obj as Partial<FunctionToolArrayProperty>

    if (type !== 'array') {
        return false
    }

    if (!['string', 'number', 'boolean'].includes(itemsType!)) {
        return false
    }

    return true
}

export type FunctionToolProperty = FunctionToolSimpleProperty | FunctionToolArrayProperty

export function isFunctionToolProperty(obj: unknown): obj is FunctionToolProperty {
    return isFunctionToolSimpleProperty(obj) || isFunctionToolArrayProperty(obj)
}

export interface FunctionToolPropertiesMap {
    [key: string]: {
        /** The JSON Schema type of the property (e.g. 'string', 'array'). */
        type: string
        /** Human-readable description of the property. */
        description?: string
        /** If type is 'array', the `items` object describing the items' type. */
        items?: string
    }
}

export function isFunctionToolPropertiesMap(obj: unknown): obj is FunctionToolPropertiesMap {
    if (!isObj(obj)) {
        return false
    }

    return Object.values(obj).every(isFunctionToolProperty)
}

export interface FunctionToolFunctionDeclaration {
    /** The name of the function. */
    name: string
    /** A description of what the function does. */
    description: string
    /** The parameters the function accepts. */
    parameters?: {
        /** Always 'object'. */
        type: 'object'
        /** Map of property definitions. */
        properties: FunctionToolPropertiesMap
        /** List of required property names. */
        required?: string[]
        /** Whether additional properties are allowed. */
        additionalProperties?: boolean
    }
    /** A flag for strict mode handling. */
    strict?: boolean
}

export function isFunctionToolFunctionDeclaration(obj: unknown): obj is FunctionToolFunctionDeclaration {
    if (!isObj(obj)) {
        return false
    }

    const { name, description, parameters, strict } = obj as Partial<FunctionToolFunctionDeclaration>

    if (!isStr(name)) {
        return false
    }

    if (!isStr(description)) {
        return false
    }

    if (isDef(parameters) && !isObj(parameters)) {
        return false
    }

    if (isDef(strict) && !isBool(strict)) {
        return false
    }

    if (isDef(parameters) && parameters.type !== 'object') {
        return false
    }

    if (isDef(parameters) && !isFunctionToolPropertiesMap(parameters.properties)) {
        return false
    }

    if (isDef(parameters) && isDef(parameters.required) && !Array.isArray(parameters.required)) {
        return false
    }

    if (
        isDef(parameters) &&
        Array.isArray(parameters.required) &&
        !parameters.required.every((requiredName) => isStr(requiredName))
    ) {
        return false
    }

    if (isDef(parameters) && isDef(parameters.additionalProperties) && !isBool(parameters.additionalProperties)) {
        return false
    }

    return true
}

export interface FunctionToolDeclaration {
    /** Will be 'function'. */
    type: 'function'
    /** The function descriptor. */
    function: FunctionToolFunctionDeclaration
}

export function isFunctionToolDeclaration(obj: unknown): obj is FunctionToolDeclaration {
    if (!isObj(obj)) {
        return false
    }

    const { type, function: func } = obj as Partial<FunctionToolDeclaration>

    if (type !== 'function') {
        return false
    }

    return isFunctionToolFunctionDeclaration(func)
}

/**
 * Parses a shorthand string description of a parameter into a typed property object.
 *
 * Shorthand format: "name : type[*]"
 * Examples: "id : string*", "tags : string[]", "count : number"
 *
 * @param paramShorthand - The shorthand string to parse.
 * @returns The parsed property object.
 * @throws TypeError If `paramShorthand` is not a string.
 * @throws SyntaxError If the input string is structurally invalid or unparseable.
 */
export function parseParamShorthand(paramShorthand: string): FunctionToolProperty {
    if (!isStr(paramShorthand)) {
        throw new TypeError(`Expected paramShorthand to be a string. Got ${paramShorthand}`)
    }

    const parts = paramShorthand.split(':')
    if (parts.length !== 2) {
        throw new SyntaxError(`Invalid paramShorthand format: ${paramShorthand}`)
    }

    const name = parts[0].trim()
    if (name.length === 0) {
        throw new SyntaxError(`Invalid paramShorthand name: ${paramShorthand}`)
    }

    const typeAndRequired = parts[1].trim()

    const required = typeAndRequired.endsWith('*')
    const type = required ? typeAndRequired.slice(0, -1).trim() : typeAndRequired.trim()
    const isArray = type.endsWith('[]')
    if (isArray) {
        const itemsType = type.slice(0, -2).trim()
        if (itemsType.length === 0) {
            throw new SyntaxError(`Invalid paramShorthand array type: ${paramShorthand}`)
        }
        return { name, type: 'array', required, itemsType }
    }
    if (type.length === 0) {
        throw new SyntaxError(`Invalid paramShorthand type: ${paramShorthand}`)
    }

    if (!['string', 'number', 'boolean'].includes(type)) {
        throw new SyntaxError(`Unsupported paramShorthand type: ${paramShorthand}`)
    }

    return { name, type, required } as FunctionToolSimpleProperty
}

/**
 * A builder class for defining tools that can be executed by LLMs.
 * Fluently define the tool name, description, parameters, and the function mapping.
 */
export class FunctionTool {
    name: string

    /** The value of `this` inside the function when it is invoked */
    thisArg: unknown = undefined

    /** An array of descriptions for each parameter the tool's function accepts. */
    properties: FunctionToolProperty[] = []

    /** The actual JavaScript function to be executed. */
    func?: (...args: unknown[]) => unknown

    /** A description of what the tool does. */
    description = ''

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
                    tool.properties.push({
                        name,
                        type: 'array',
                        description: prop.description,
                        required,
                        itemsType: prop.items!,
                    } as FunctionToolArrayProperty)
                } else {
                    tool.properties.push({
                        name,
                        type: prop.type,
                        description: prop.description,
                        required,
                    } as FunctionToolSimpleProperty)
                }
            }
            tool.hasAdditionalProperties(declaration.function.parameters.additionalProperties ?? false)
        }
        tool.strictMode(declaration.function.strict ?? false)
        return tool
    }

    constructor(name: string, ...description: string[]) {
        if (!isStr(name)) {
            throw new TypeError(`Expected tool name to be a string. Got ${name}`)
        }
        this.name = name
        this.description = description.join(' ')
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
            throw new TypeError(`Expected tool func to be a function. Got ${func}`)
        }
        this.func = func
        return this
    }

    /**
     * Sets the `this` context to be used when the tool's function is invoked.
     *
     * Note: To preserve runtime compatibility we define it as a computed property name so callers that
     * access the property via bracket-notation (e.g. `tool['this'](...)`) will continue to work.
     *
     * @param thisArg The object to use as the `this` context.
     * @returns The tool instance for chaining.
     */
    ['this'](thisArg: unknown): this {
        this.thisArg = thisArg
        return this
    }

    /**
     * Indicates whether the underlying function accepts parameters not explicitly defined.
     *
     * @param value True to allow additional properties, false otherwise.
     * @returns The tool instance for chaining.
     */
    hasAdditionalProperties(value: boolean): this {
        this.additionalProperties = Boolean(value)
        return this
    }

    /**
     * Sets whether strict mode tracking should be enabled. Often used for JSON schema validation enforcements by LLMs.
     *
     * @param value True to enforce strict mode, false otherwise.
     * @returns The tool instance for chaining.
     */
    strictMode(value: boolean): this {
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
            return `Error executing ${this.name}(${argsStr}): ${error}`
        }
    }

    /**
     * Reusable builder step to define a tool argument/parameter via the shorthand format.
     *
     * @param paramShorthand Shorthand describing the param formatting (e.g., "id: string*").
     * @param description An explanation of the parameter's intent.
     * @returns The tool instance for chaining.
     */
    prm(paramShorthand: string, description?: string): this {
        const parsedParam = parseParamShorthand(paramShorthand)
        this.properties.push({ ...parsedParam, description })
        return this
    }

    /**
     * Gets the tool's description in a format suitable for LLM function calling.
     * @returns The structured description of the tool.
     */
    toJSON(): FunctionToolDeclaration {
        const properties: FunctionToolPropertiesMap = {}
        const required: string[] = []
        for (const p of this.properties) {
            properties[p.name] = {
                type: p.type,
                description: p.description,
            }

            if (p.required) {
                required.push(p.name)
            }

            if (isFunctionToolArrayProperty(properties[p.name])) {
                properties[p.name].items = (p as FunctionToolArrayProperty).itemsType
            }
        }

        return {
            type: 'function',
            function: {
                name: this.name,
                description: this.description,
                parameters: {
                    type: 'object',
                    properties,
                    required,
                    additionalProperties: this.additionalProperties,
                },
                strict: this.strict,
            },
        }
    }
}
