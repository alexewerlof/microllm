import { Message } from '@huggingface/transformers'
import { FunctionToolDeclaration } from '../Tools/types.js'
import { SupportedMessage } from '../Message/types.js'

/**
 * Input to the adapter hook that runs before tokenizer chat templating.
 */
export interface BeforeChatTemplateParams {
    /** Conversation history in MicroLLM's canonical message format. */
    messages: SupportedMessage[]
    /** Optional function tools available for this turn. */
    tools?: FunctionToolDeclaration[]
}

/**
 * Output from the adapter hook that prepares chat template inputs.
 */
export interface BeforeChatTemplateResult {
    /** Model-ready message history passed to tokenizer.apply_chat_template(). */
    messages: Message[]
    /** Model-specific template options merged into apply_chat_template(). */
    templateOptions: Record<string, unknown>
}

/**
 * Input to the adapter hook that runs after text is decoded.
 */
export interface AfterDecodeParams {
    /** Decoded assistant text preserving special tokens. */
    rawAssistantContent: string
    /** Decoded assistant text with special tokens removed. */
    cleanAssistantContent: string
}

/**
 * Lifecycle hooks that isolate model-family semantics from the core chat loop.
 *
 * @example
 * ```ts
 * const adapter: ChatModelAdapter = {
 *   name: 'my-adapter',
 *   onBeforeChatTemplate: ({ messages, tools }) => ({
 *     messages: messages as unknown as Message[],
 *     templateOptions: { tools },
 *   }),
 *   onAfterDecode: ({ cleanAssistantContent }) => ({
 *     role: 'assistant',
 *     content: cleanAssistantContent,
 *   }),
 * }
 * ```
 */
export interface ChatModelAdapter {
    /** Human-readable adapter name used in diagnostics. */
    name: string

    /**
     * Converts canonical messages into model-ready messages and template options.
     */
    onBeforeChatTemplate(params: BeforeChatTemplateParams): BeforeChatTemplateResult

    /**
     * Interprets a decoded assistant response and returns a canonical message.
     */
    onAfterDecode(params: AfterDecodeParams): SupportedMessage
}
