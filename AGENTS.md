This repository contains a small & lightweight LLM library using [Liquid.ai](https://docs.liquid.ai) and [@huggingface/transformers.js](https://huggingface.co/docs/transformers.js).

# Code readability

- Use an empty line to separate different concerns in a block of code
- The library source code is in the `src` directory and is written in idiomatic TypeScript using the latest version of the language.
- Package different parts of the algorithm in small pure functions that are easy to test and have expressive names. A good rule of thumb is when the function does not have too many variables and its body is less than 10 lines of code.
- Each function should be documented using TypeDocs format with `@example` block that shows how it is supposed to be used.
- The API surface should follow the fluent chainable format which makes the user code small, focused and expressive.
- Use expressive variable names and don't worry about the length of internal variable names.
- Only use `class` for stateful algorithms or cases where the book keeping of an otherwise parameter is too much of a hassle for the user.
- Avoid inheritance as much as possible. When `extend`ing a class, it should be about adding a new method not to override it.
- Instead of hard coding values that aren't obvious (e.g. '<|tool_call_start|>'), define const variables with expressive names like `TOOL_CALL_TOKEN` and use that instead.

# Defensive programming

- Never assume the shape, content, or types of a value that is coming from outside this code base: user code, dependencies, databases, AI generated, APIs, etc. Always verify.
- Use the `jty` library extensively to validate the types that you expect because although the code is written in TypeScript, it may be called with Javascript where typechecking is absent. Also when working with the Small Language Models we cannot be sure that they follow all the instructions. Always verify and fail with meaningful error messages that help you debug the issue and fix it.

# Tests

- We use Node.js native testing framework
- Each function is tested in its own `describe()` block
- Instead of hard coding the function name, use the `.name` property of a function. That way renaming the function in the code, keeps the describe clause up to date:
    - BAD: `describe('tryParseToolCalls',`
    - GOOD: `describe(tryParseToolCalls.name,`
- The first few `test()` blocks test the expected behavior (happy case)
- They are followed by more `test()` blocks which test different edge cases in the order of how common you believe they can be.
- Larget algorithms are implemented in a series of functions that depend on each other. Always start the tests from the leaf nodes of that dependency tree to ensure the logic bottom up.

## \_test convention

- Every module has a public API that is intended to be used by the other modules, and optionally some internal constructs (e.g. function, class) that are used for internal implementation.
- Only the public API should be `export`ed directly. All internal constructs must NOT be exported directly — they should be exposed exclusively through `export const _test = { ... }`.
- This gives consumers of the module a clear signal: direct exports are the stable public API, while anything behind `_test` is an implementation detail subject to change.
- In tests, import only `_test` and the public API. Destructure `_test` at the top of the test file for convenience. Use `_test.fnName.name` in `describe()` blocks so the test name stays in sync with the code.

For example:

```ts
// Internal implementation detail — NOT exported directly
function getId(obj: { id: string }): string {
    return obj.id
}

// Public API — exported directly
export function getChunkIds(chunks: { id: string; content: string }[]): string[] {
    return chunks.map(getId)
}

// Internal constructs exposed for testing only
export const _test = { getId }
```

Then test it like this:

```ts
import { _test, getChunkIds } from '...'

const { getId } = _test

describe(_test.getId.name, () => {
    // Test the internal function
})
describe(getChunkIds.name, () => {
    // Test the public API
})
```
