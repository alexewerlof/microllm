import { ProgressCallback, ProgressInfo } from '@huggingface/transformers'
import { SingleBar } from 'cli-progress'
import { bytesToHumanReadable } from '../src/utilities/format.js'

export function createProgressCallback(task: string = ''): ProgressCallback {
    const progressBar = new SingleBar({
        format: `${task} pipeline {loadedHuman} of {totalHuman} {bar} {percentage}%`,
        fps: 10,
        barCompleteChar: '\u2588',
        barIncompleteChar: '\u2591',
    })

    let isInitialized = false

    return (p: ProgressInfo) => {
        switch (p.status) {
            case 'progress_total':
                {
                    const payload = {
                        ...p,
                        loadedHuman: bytesToHumanReadable(p.loaded),
                        totalHuman: bytesToHumanReadable(p.total),
                    }
                    if (isInitialized) {
                        progressBar.update(p.loaded, payload)
                    } else {
                        progressBar.start(p.total, p.loaded, payload)
                        isInitialized = true
                    }
                }
                break
            case 'ready':
                progressBar.stop()
                break
        }
    }
}
