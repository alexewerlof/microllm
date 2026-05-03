import { hasProp, inArr, isArr, isArrLen, isDef, isInt, isObj, isPOJO, isStr, isStrLen } from 'jty'
import { isFunctionToolDeclaration } from './Tools/index.js'
import { SupportedMessage } from './Message/types.js'
import { MicroLLM, MicroLLMCompleteParams } from './MicroLLM.js'
import { isSupportedMessage } from './Message/guards.js'

const SUPPORTED_HTTP_METHODS = ['GET', 'POST'] as const

export interface MicroAPICompletionConfig {
    model?: string
    temperature?: number
    top_p?: number
    max_tokens?: number
    frequency_penalty?: number
    presence_penalty?: number
    stop?: string | string[]
    [key: string]: unknown
}

export interface MicroAPIOptions {
    baseUrl: string
    modelId: string
    apiKey?: string
    useApiKey?: boolean
    defaultConfig?: Partial<MicroAPICompletionConfig>
    extraHeaders?: Record<string, string>
}

export interface ChatCompletionsChoice {
    index: number
    message: SupportedMessage
    finish_reason: string
}

export function isChatCompletionsChoice(x: unknown): x is ChatCompletionsChoice {
    return (
        isObj(x) &&
        hasProp(x, 'index', 'message', 'finish_reason') &&
        isInt(x.index) &&
        isSupportedMessage(x.message) &&
        isStr(x.finish_reason)
    )
}

export interface ChatCompletionsUsage {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
}

export function isChatCompletionsUsage(x: unknown): x is ChatCompletionsUsage {
    return (
        isObj(x) &&
        hasProp(x, 'prompt_tokens', 'completion_tokens', 'total_tokens') &&
        isInt(x.prompt_tokens) &&
        isInt(x.completion_tokens) &&
        isInt(x.total_tokens)
    )
}

export interface ChatCompletionsResponse {
    id: string
    object: string
    created: number
    model: string
    choices: ChatCompletionsChoice[]
    usage?: ChatCompletionsUsage
}

export function isChatCompletionsResponse(x: unknown): x is ChatCompletionsResponse {
    return (
        isObj(x) &&
        hasProp(x, 'id', 'object', 'created', 'model', 'choices') &&
        isStr(x.id) &&
        isStr(x.object) &&
        isInt(x.created) &&
        isStr(x.model) &&
        isArrLen(x.choices, 1) &&
        x.choices.every(isChatCompletionsChoice) &&
        (!hasProp(x, 'usage') || isChatCompletionsUsage(x.usage))
    )
}

/**
 * OpenAI-compatible chat client for local or hosted LLM APIs.
 */
export class MicroAPI implements MicroLLM<MicroAPICompletionConfig> {
    baseUrl: URL
    modelId: string
    apiKey: string
    useApiKey: boolean
    defaultConfig: Partial<MicroAPICompletionConfig>
    isBusy = false
    extraHeaders: Record<string, string>

    /**
     * Creates a new OpenAI-compatible API client.
     */
    constructor(options: MicroAPIOptions) {
        if (!isPOJO(options)) {
            throw new TypeError(`Expected object for options, but got ${options} (${typeof options})`)
        }

        const { baseUrl, modelId, apiKey = '', useApiKey = false, defaultConfig = {}, extraHeaders = {} } = options

        if (!isStrLen(modelId, 1)) {
            throw new TypeError(`Expected non-empty string for modelId, but got ${modelId} (${typeof modelId})`)
        }

        if (!isStr(apiKey)) {
            throw new TypeError(`Expected string for apiKey, but got ${apiKey} (${typeof apiKey})`)
        }

        if (!isPOJO(defaultConfig)) {
            throw new TypeError(
                `Expected plain object for defaultConfig, but got ${defaultConfig} (${typeof defaultConfig})`,
            )
        }

        if (!isPOJO(extraHeaders)) {
            throw new TypeError(
                `Expected plain object for extraHeaders, but got ${extraHeaders} (${typeof extraHeaders})`,
            )
        }

        this.baseUrl = new URL(baseUrl)
        this.modelId = modelId
        this.apiKey = apiKey
        this.useApiKey = Boolean(useApiKey)
        this.defaultConfig = defaultConfig
        this.extraHeaders = extraHeaders
    }

    /**
     * Fetches the list of available models from the API.
     */
    async getModels(): Promise<unknown[]> {
        const response = await this.fetchJson('GET', 'models')

        if (!isObj(response) || !hasProp(response, 'data') || !isArr(response.data)) {
            throw new TypeError(`Expected models response to expose data array, but got ${JSON.stringify(response)}`)
        }

        return response.data as unknown[]
    }

    /**
     * Fetches and returns a sorted list of model IDs.
     */
    async getModelIds(): Promise<string[]> {
        const models = await this.getModels()
        return models
            .filter((model): model is { id: string } => isObj(model) && hasProp(model, 'id') && isStr(model.id))
            .map((model) => model.id)
            .sort()
    }

    /**
     * Completes a thread using the configured OpenAI-compatible endpoint.
     */
    async complete(params: MicroLLMCompleteParams<Partial<MicroAPICompletionConfig>>): Promise<SupportedMessage> {
        if (!isObj(params)) {
            throw new TypeError(`Expected object for params, but got ${params} (${typeof params})`)
        }

        const { messages, config, signal, tools } = params

        if (!isArr(messages)) {
            throw new TypeError(`Expected array for messages, but got ${messages} (${typeof messages})`)
        }

        if (isDef(config) && !isPOJO(config)) {
            throw new TypeError(`Expected plain object for config, but got ${config} (${typeof config})`)
        }

        if (isDef(tools) && (!isArr(tools) || !tools.every(isFunctionToolDeclaration))) {
            throw new TypeError(
                `Expected tools to be an array of FunctionToolDeclaration objects, but got ${JSON.stringify(tools, null, 2)} (${typeof tools})`,
            )
        }

        const requestConfig = {
            ...this.defaultConfig,
            ...config,
        }

        const response = await this.fetchJson(
            'POST',
            'chat/completions',
            {
                messages,
                model: requestConfig.model ?? this.modelId,
                tools,
                ...requestConfig,
            },
            signal,
        )

        if (!isChatCompletionsResponse(response)) {
            throw new TypeError(
                `Expected response to conform to ChatCompletionsResponse, but got ${JSON.stringify(response, null, 2)}`,
            )
        }

        return response.choices[0].message
    }

    private makeUrl(path: string): URL {
        const { href } = this.baseUrl
        return new URL(path, href.endsWith('/') ? href : `${href}/`)
    }

    private createHeaders(method: string): Headers {
        const headers = new Headers()
        headers.set('Accept', 'application/json')
        headers.set('Accept-Charset', 'utf-8')
        headers.set('Connection', 'keep-alive')

        if (method === 'POST') {
            headers.set('Content-Type', 'application/json')
        }

        if (this.baseUrl.href.toLowerCase().includes('api.anthropic.com')) {
            headers.set('anthropic-dangerous-direct-browser-access', 'true')
        }

        if (this.useApiKey) {
            if (!isStrLen(this.apiKey, 1)) {
                throw new Error('API key is not set, but useApiKey is enabled')
            }

            headers.set('Authorization', `Bearer ${this.apiKey}`)
        }

        for (const [key, value] of Object.entries(this.extraHeaders)) {
            if (isStrLen(key, 1) && isStr(value)) {
                headers.set(key, value)
            }
        }

        return headers
    }

    private async fetchJson(method: string, path: string, data?: unknown, signal?: AbortSignal): Promise<unknown> {
        const methodUpperCase = method.toUpperCase()

        if (!inArr(methodUpperCase, [...SUPPORTED_HTTP_METHODS])) {
            throw new RangeError(`Unsupported HTTP method: ${method}`)
        }

        if (methodUpperCase === 'POST' && !isObj(data)) {
            throw new TypeError(`POST data must be an object. Got ${data} (${typeof data})`)
        }

        const url = this.makeUrl(path)
        const headers = this.createHeaders(methodUpperCase)

        try {
            this.isBusy = true

            const response = await fetch(url, {
                method: methodUpperCase,
                headers,
                signal,
                body: methodUpperCase === 'POST' ? JSON.stringify(data) : undefined,
            })

            if (!response.ok) {
                const errorBody = await response.text()
                throw new Error(`HTTP ${response.status} ${methodUpperCase} ${url}\n${errorBody}`)
            }

            return await response.json()
        } catch (cause) {
            throw new Error(`Failed ${methodUpperCase} request to ${url}`, { cause })
        } finally {
            this.isBusy = false
        }
    }
}
