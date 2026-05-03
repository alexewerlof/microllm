import { hasProp, isFn, isObj } from 'jty'
import { SupportedMessage } from './Message/types.js'
import { FunctionToolDeclaration } from './Tools/types.js'

/**
 * Generic completion input contract shared by all MicroLLM implementations.
 */
export interface MicroLLMCompleteParams<TConfig> {
    /** The conversation messages. */
    messages: SupportedMessage[]
    /** Optional model-specific completion config. */
    config?: TConfig
    /** Optional abort signal. */
    signal?: AbortSignal
    /** Optional tool declarations available during completion. */
    tools?: FunctionToolDeclaration[]
    /** Optional callback invoked for generated tokens when supported. */
    onToken?: (token: string) => unknown
}

/**
 * Minimal contract for any chat-capable language model backend.
 */
export interface MicroLLM<TConfig = unknown> {
    /**
     * Completes a message thread and returns a supported assistant message shape.
     */
    complete(params: MicroLLMCompleteParams<TConfig>): Promise<SupportedMessage>
}

/**
 * Checks whether a value exposes the structural shape of a MicroLLM.
 *
 * @example
 * ```ts
 * isMicroLLM({ complete: async () => ({ role: 'assistant', content: 'ok' }) })
 * ```
 */
export function isMicroLLM(x: unknown): x is MicroLLM<unknown> {
    return isObj(x) && hasProp(x, 'complete') && isFn(x.complete)
}
