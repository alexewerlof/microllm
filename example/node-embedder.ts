import { MicroEmbedder } from "../src/MicroEmbedder";

async function main() {
    const microEmbedder = new MicroEmbedder('Xenova/all-MiniLM-L6-v2')
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

main().catch(console.error)