import { MicroEmbedder } from "../src/MicroEmbedder";
import { PipelineFactory } from "../src/PipelineFactory";

async function main() {
    const pipelineFactory = new PipelineFactory('feature-extraction', 'Xenova/all-MiniLM-L6-v2')
    const microEmbedder = new MicroEmbedder(pipelineFactory)
    const sentences = [
        "Hello, how are you?",
        "What is the capital of France?",
        "I love machine learning."
    ]

    async function printEmbedding(text: string) {
        const embedding = await microEmbedder.embed(text)
        console.log(`Embedding for "${text}":`, embedding)
    }
    await Promise.all(sentences.map(printEmbedding))
    console.log('All embeddings generated.')
}

main().catch(error => {
    console.error('An error occurred:', error)
    process.exit(1)
})