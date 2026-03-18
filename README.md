# MicroLLM

**Minimalistic Small Language Model Library for JavaScript**

MicroLLM is a lightweight, bare-bones library designed to run Small Language Models (SLMs) effortlessly in JavaScript. Whether you're working in Node.js (CPU) or the browser (WASM, WebGPU), MicroLLM provides a unified codebase to power your AI applications.

## Why MicroLLM?

- **Cross-Platform Support**: Seamlessly run on Node.js (CPU) and browsers (WASM, WebGPU).
- **TypeScript Ready**: Includes TypeScript types out of the box for a smooth development experience.
- **OpenAI-Compatible API**: Minimal to no refactoring required to integrate with your existing OpenAI-based projects.
- **Cost-Effective**: No need for expensive GPUs—run LLMs efficiently on your existing hardware.
- **Learn by Example**: A rich collection of examples to help you get started quickly.

## Notable Features

### MicroChat
Engage in conversations with an instruct model, enabling dynamic and interactive AI-driven dialogues.

### MicroAgent
Leverage tool-calling capabilities to extend the functionality of your AI workflows.

### MicroRAG
Implement Retrieval-Augmented Generation (RAG) with ease using:
- Header-based chunking
- MicroEmbedder for embeddings
- In-memory VectorStore for efficient retrieval

## What Problems Does MicroLLM Solve?

- **Cost Savings**: Stop paying AI vendors for your development and experimentation workloads.
- **Learn AI Systems Engineering**: Master the art of creating small language models that can interact with each other.
- **Privacy and Experimentation**: Like `MicroK8s` compared to full-blown `Kubernetes`, MicroLLM offers small, performant models that enable experimentation without the cost or privacy concerns of sending data to AI vendors.
- **Production-Ready**: MIT licensed, making it perfect for shipping to production for simple tasks like:
  - [AI Firewall](https://blog.alexewerlof.com/p/ai-firewall)
  - Agentic workloads
  - Client-side FAQs

## Implementation Details

- **Powered by Hugging Face**: Built on [@huggingface/transformers.js](https://www.npmjs.com/package/@huggingface/transformers), which utilizes the [ONNX](https://onnxruntime.ai/) runtime.
- **Customizable**: Unlike other libraries, MicroLLM doesn't abstract away Hugging Face, allowing you to tweak settings as needed.
- **Automatic Caching**: Handles caching seamlessly in both Node.js and browser environments.
- **Model Support**: Currently supports the latest LiquidFM, with plans to add more small models in the future.

## Installation

Install MicroLLM via npm:

```bash
npm i microllm
```

## Usage

Explore the [examples](examples/) directory to see MicroLLM in action:
- Node.js implementations
- Browser-based demos
- Retrieval-Augmented Generation (RAG)
- Tool calling and more!

Run the node examples using `tsx` for example:

```bash
node --loader tsx examples/node-rag.ts 
```

## Contributing

We welcome contributions! Feel free to submit issues, feature requests, or pull requests to help improve MicroLLM.

## License

This project is licensed under the MIT License.

---

🇸🇪 Made in Sweden by [Alex Ewerlof](https://alexewerlof.com)