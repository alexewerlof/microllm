/**
 * The common base properties for defining a function tool parameter (simple or array).
 */
export interface FunctionToolBaseProperty {
    /** The data type of the parameter (e.g., 'string', 'number', 'boolean'). */
    type: string
    /** Whether this parameter is required. */
    required?: boolean
    /** A human-readable description of the parameter. */
    description?: string
}

export const SUPPORTED_SIMPLE_TYPES = ['string', 'number', 'boolean'] as const

export type SupportedSimpleType = (typeof SUPPORTED_SIMPLE_TYPES)[number]

export interface FunctionToolSimpleProperty extends FunctionToolBaseProperty {
    type: SupportedSimpleType
}

export interface FunctionToolArrayProperty extends FunctionToolBaseProperty {
    type: 'array'
    /** When `type` is 'array', this specifies the type of the items in the array (e.g., 'string'). */
    itemsType: SupportedSimpleType
}

export type FunctionToolProperty = FunctionToolSimpleProperty | FunctionToolArrayProperty

export interface FunctionToolProperties {
    [key: string]: FunctionToolProperty
}

export interface FunctionToolDeclarationParameters {
    /** Always 'object'. */
    type: 'object'
    /** Map of property definitions. */
    properties: FunctionToolProperties
    /** List of required property names. */
    required?: string[]
    /** Whether additional properties are allowed. */
    additionalProperties?: boolean
}

export interface FunctionToolDeclarationFunction {
    /** The name of the function. */
    name: string
    /** A description of what the function does. */
    description: string
    /** The parameters the function accepts. */
    parameters?: FunctionToolDeclarationParameters
    /** A flag for strict mode handling. */
    strict?: boolean
}

export interface FunctionToolDeclaration {
    /** Will be 'function'. */
    type: 'function'
    /** The function descriptor. */
    function: FunctionToolDeclarationFunction
}
