import { ProgressCallback, ProgressInfo } from "@huggingface/transformers"
import cliProgress from 'cli-progress'

export function createProgressCallback(task: string = ''): ProgressCallback {
    const cliProgressContainer = new cliProgress.MultiBar({
        format: task + ` {bar} | {name} | {file} | {percentage} | Remaining: {eta_formatted}`,
        stopOnComplete: true,
    }, cliProgress.Presets.shades_grey);

    const files: {
        [name: string]: {
            [file: string]: cliProgress.SingleBar
        }
    } = {}

    function getBar(name: string, file: string): cliProgress.SingleBar {
        if (!files[name]) {
            files[name] = {}
        }
        if (!files[name][file]) {
            files[name][file] = cliProgressContainer.create(100, 0, { name, file })
        }
        return files[name][file]
    }

    return (progressInfo: ProgressInfo) => {
        switch (progressInfo.status) {
            case 'initiate':
                getBar(progressInfo.name, progressInfo.file).start(100, 0, progressInfo)
                break
            case 'download':
                getBar(progressInfo.name, progressInfo.file).start(100, 0, progressInfo)
                break
            case 'progress':
                getBar(progressInfo.name, progressInfo.file).update(progressInfo.progress, progressInfo)
                break
            case 'done':
                {
                    const bar = getBar(progressInfo.name, progressInfo.file)
                    bar.update(bar.getTotal(), progressInfo)
                    bar.stop()
                }
                break
            case 'ready':
                cliProgressContainer.stop()
                break
            default:
                console.warn(`Progress event with unknown status: ${JSON.stringify(progressInfo)}`)
        }
    };
}