import { hasProp, isFn, isObj } from 'jty'
import { ChatModelAdapter } from './types.js'

/**
 * Checks whether a value matches the structural shape of a chat model adapter.
 *
 * @param x The value to validate.
 * @returns True when the value exposes the required adapter hooks.
 *
 * @example
 * ```ts
 * isChatModelAdapter({ onBeforeChatTemplate() {}, onAfterDecode() {} })
 * ```
 */
export function isChatModelAdapter(x: unknown): x is ChatModelAdapter {
    return isObj(x) && hasProp(x, 'onBeforeChatTemplate', 'onAfterDecode') && isFn(x.onBeforeChatTemplate) && isFn(x.onAfterDecode)
}