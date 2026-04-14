import { DataType, DeviceType, ModelRegistry, PretrainedModelOptions } from '@huggingface/transformers'
import { confirm, input, select, Separator } from '@inquirer/prompts'
import { createProgressCallback } from './progress-callback.js'
import { PipelineFactory } from '../src/PipelineFactory.js'

export async function inquirePipeline() {
    let modelId = await select({
        message: 'Select a package manager',
        choices: [
            {
                name: 'Liquid 2 1.2B',
                value: 'onnx-community/LFM2-1.2B-ONNX',
                description: 'https://huggingface.co/onnx-community/LFM2-1.2B-ONNX',
            },
            {
                name: 'Liquid 2.5 1.2B Thinking',
                value: 'LiquidAI/LFM2.5-1.2B-Thinking-ONNX',
                description: 'https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-ONNX',
            },
            {
                name: 'Liquid 2 1.2B RAG',
                value: 'onnx-community/LFM2-1.2B-RAG-ONNX',
                description: 'https://huggingface.co/onnx-community/LFM2-1.2B-RAG-ONNX',
            },
            {
                name: 'Liquid 2 1.2B Tool Calling',
                value: 'onnx-community/LFM2-1.2B-Tool-ONNX',
                description: 'https://huggingface.co/onnx-community/LFM2-1.2B-Tool-ONNX',
            },
            new Separator(),
            {
                name: 'Gemma 4 E4B Instruction Tuned',
                value: 'onnx-community/gemma-4-E4B-it-ONNX',
                description: 'https://huggingface.co/onnx-community/gemma-4-E4B-it-ONNX',
            },
            {
                name: 'Gemma 3 1B Instruction Tuned',
                value: 'onnx-community/gemma-3-1b-it-ONNX',
                description: 'https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX',
                disabled: true,
            },
            {
                name: 'Custom',
                value: '',
                description: 'Enter a custom model id from Hugging Face',
            },
        ],
    })

    if (!modelId) {
        modelId = await input({
            message: 'Enter a model id from Hugging Face Model Hub (format: ORG/REPO or ORG/REPO@REVISION)',
            validate(value) {
                if (!/^[-\w]+\/[-\w]+(@[\w\d.\-_]+)?$/.test(value)) {
                    return 'Invalid model id format. Expected ORG/REPO or ORG/REPO@REVISION'
                }
                return true
            },
        })
    }

    console.log(`Model Id: ${modelId}! Checking available quantizations...`)

    const dtypes = await ModelRegistry.get_available_dtypes(modelId)

    const dtype = (await select({
        message: 'Select a quantization',
        choices: dtypes.map((dtype) => ({
            name: dtype,
            value: dtype,
        })),
    })) as DataType

    console.log(`Dtype: ${dtype}! Checking cache...`)

    const isCached = await ModelRegistry.is_cached(modelId, { dtype })

    let shouldDownload = true
    if (isCached) {
        const useCache = await confirm({
            message: 'Use cached model?',
            default: true,
        })
        if (useCache) {
            shouldDownload = false
        } else {
            await ModelRegistry.clear_cache(modelId, { dtype })
        }
    }

    const device = (await select({
        message: 'Select a device',
        pageSize: 20,
        default: 'auto',
        choices: [
            new Separator('── Auto ──'),
            { name: 'Auto', value: 'auto', description: 'Auto-detect best device for the environment' },
            {
                name: 'GPU',
                value: 'gpu',
                description: 'Auto-select best available GPU (CUDA / WebGPU / DirectML / WebNN-GPU)',
            },
            { name: 'CUDA', value: 'cuda', description: 'NVIDIA GPU (Linux x64 only)' },
            { name: 'WebGPU', value: 'webgpu', description: 'GPU via WebGPU API' },
            { name: 'CoreML', value: 'coreml', description: 'Apple Neural Engine (macOS only)' },
            { name: 'DirectML', value: 'dml', description: 'DirectML GPU (Windows only)' },
            { name: 'CPU', value: 'cpu', description: 'Default for Node.js' },
            { name: 'WebNN', value: 'webnn', description: 'WebNN — default backend' },
            { name: 'WebNN NPU', value: 'webnn-npu', description: 'WebNN Neural Processing Unit' },
            { name: 'WebNN GPU', value: 'webnn-gpu', description: 'WebNN GPU backend' },
            { name: 'WebNN CPU', value: 'webnn-cpu', description: 'WebNN CPU backend' },
        ],
    })) as DeviceType

    console.log(`Device: ${device}!`)

    const task = 'text-generation'
    const pipelineOptions: PretrainedModelOptions = {
        dtype,
        device,
        progress_callback: createProgressCallback(task),
    }

    const p = new PipelineFactory(task, modelId, pipelineOptions)

    if (
        shouldDownload &&
        (await confirm({
            message: isCached ? 'Load model?' : 'Download model?',
            default: true,
        }))
    ) {
        await p.getPipeline()
    }
    return p
}
