import { defineConfig, type Options } from 'tsup'

const baseConfig: Omit<Options, 'entry' | 'minify' | 'platform'> = {
    bundle: true,
    outDir: 'lib',
    dts: true,
    sourcemap: true,
    tsconfig: 'tsconfig.build.json',
    noExternal: ['jty', '@huggingface/transformers'],
}

const nodeConfig: Omit<Options, 'entry' | 'minify'> = {
    ...baseConfig,
    platform: 'node',
    format: ['esm', 'iife', 'cjs'],
}

const browserConfig: Omit<Options, 'entry' | 'minify'> = {
    ...baseConfig,
    platform: 'browser',
    format: ['esm', 'iife'],
}

export default defineConfig([
    {
        entry: { bundle: 'src/index.ts' },
        minify: false,
        ...nodeConfig,
    },
    {
        entry: { 'bundle.min': 'src/index.ts' },
        minify: true,
        ...nodeConfig,
    },
    {
        entry: { 'bundle.browser': 'src/index.ts' },
        minify: false,
        ...browserConfig,
    },
    {
        entry: { 'bundle.browser.min': 'src/index.ts' },
        minify: true,
        ...browserConfig,
    },
])
