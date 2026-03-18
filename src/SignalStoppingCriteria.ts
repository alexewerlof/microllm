import { StoppingCriteria, StoppingCriteriaList } from '@huggingface/transformers'
import { isA } from 'jty'

/**
 * Custom stopping criteria that halts LLM text generation when an AbortSignal is specifically triggered.
 * Extends Hugging Face's `StoppingCriteria`.
 */
export class SignalStoppingCriteria extends StoppingCriteria {
    #signal: AbortSignal

    /**
     * Initializes the stopping criteria with a specific abort signal.
     *
     * @param {AbortSignal} signal The signal to listen to for cancellations.
     * @throws {TypeError} If the provided signal is not an instance of AbortSignal.
     */
    constructor(signal: AbortSignal) {
        super()
        if (!isA(signal, AbortSignal)) {
            throw new TypeError(`Expected signal to be an AbortSignal, but got ${signal} (${typeof signal})`)
        }
        this.#signal = signal
    }

    /**
     * Called by the generation pipeline to determine if generation should be stopped.
     *
     * @param {unknown[]} input_ids The current token sequence IDs generated so far.
     * @returns {boolean[]} An array of booleans indicating whether generation should stop for each sequence.
     */
    _call(input_ids: unknown[]): boolean[] {
        return new Array(input_ids.length).fill(this.#signal.aborted)
    }

    static createStoppingCriteriaList(signal: AbortSignal): StoppingCriteriaList {
        if (!isA(signal, AbortSignal)) {
            throw new TypeError(`Expected signal to be an AbortSignal, but got ${signal} (${typeof signal})`)
        }
        const stoppingCriteriaList = new StoppingCriteriaList()
        stoppingCriteriaList.push(new SignalStoppingCriteria(signal))
        return stoppingCriteriaList
    }
}
