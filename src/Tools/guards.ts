import { isArr, isBool, isDef, isObj, isStr } from 'jty'
import { FunctionToolArrayProperty, FunctionToolBaseProperty, FunctionToolDeclaration, FunctionToolDeclarationFunction, FunctionToolDeclarationParameters, FunctionToolProperties, FunctionToolProperty, FunctionToolSimpleProperty, SUPPORTED_SIMPLE_TYPES, SupportedSimpleType } from './types.js'

function isFunctionToolBaseProperty(obj: unknown): obj is FunctionToolBaseProperty {
    if (!isObj(obj)) {
        return false
    }

    const { type, required, description } = obj as Partial<FunctionToolBaseProperty>

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


export function isSupportedSimpleType(value: unknown): value is SupportedSimpleType {
    return isStr(value) && SUPPORTED_SIMPLE_TYPES.includes(value as SupportedSimpleType)
}

function isFunctionToolSimpleProperty(obj: unknown): obj is FunctionToolSimpleProperty {
    return isFunctionToolBaseProperty(obj) && isSupportedSimpleType(obj.type)
}

function isFunctionToolArrayProperty(obj: unknown): obj is FunctionToolArrayProperty {
    if (!isFunctionToolBaseProperty(obj)) {
        return false
    }

    const { type, itemsType } = obj as Partial<FunctionToolArrayProperty>

    if (type !== 'array') {
        return false
    }

    return isSupportedSimpleType(itemsType)
}

function isFunctionToolProperty(obj: unknown): obj is FunctionToolProperty {
    return isFunctionToolSimpleProperty(obj) || isFunctionToolArrayProperty(obj)
}

function isFunctionToolProperties(obj: unknown): obj is FunctionToolProperties {
    if (!isObj(obj)) {
        return false
    }

    return Object.values(obj).every(isFunctionToolProperty)
}

export function isFunctionToolDeclarationParameters(obj: unknown): obj is FunctionToolDeclarationParameters {
    if (!isObj(obj)) {
        return false
    }

    const { type, properties, required, additionalProperties } = obj as Partial<FunctionToolDeclarationParameters>

    if (type !== 'object') {
        return false
    }

    if (isDef(required) && !isArr(required)) {
        return false
    }

    if (isDef(additionalProperties) && !isBool(additionalProperties)) {
        return false
    }

    return isFunctionToolProperties(properties)
}

export function isFunctionToolDeclarationFunction(obj: unknown): obj is FunctionToolDeclarationFunction {
    if (!isObj(obj)) {
        return false
    }

    const { name, description, strict, parameters } = obj as Partial<FunctionToolDeclarationFunction>

    if (!isStr(name) || !isStr(description)) {
        return false
    }

    if (isDef(strict) && !isBool(strict)) {
        return false
    }

    if (isDef(parameters) && !isFunctionToolDeclarationParameters(parameters)) {
        return false
    }

    return true
}

export function isFunctionToolDeclaration(obj: unknown): obj is FunctionToolDeclaration {
    if (!isObj(obj)) {
        return false
    }

    const { type, function: func } = obj as Partial<FunctionToolDeclaration>

    if (type !== 'function') {
        return false
    }

    return isFunctionToolDeclarationFunction(func)
}