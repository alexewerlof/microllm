This repository contains a small & lightweight LLM library using [Liquid.ai](https://docs.liquid.ai) and [@huggingface/transformers.js](https://huggingface.co/docs/transformers.js).

# Source code

- Use an empty line to separate different concerns in a block of code
- The library source code is in the `src` directory and is written in idiomatic TypeScript using the latest version of the language.
- Package different parts of the algorithm in small pure functions that are easy to test and have expressive names. A good rule of thumb is when the function does not have too many variables and its body is less than 10 lines of code.
- Each function should be documented using TypeDocs format with `@example` block that shows how it is supposed to be used.
- The API surface should follow the fluent chainable format which makes the user code small, focused and expressive.
- Use expressive variable names and don't worry about the length of internal variable names.
- Only use `class` for stateful algorithms or cases where the book keeping of an otherwise parameter is too much of a hassle for the user.
- Avoid inheritance as much as possible. When `extend`ing a class, it should be about adding a new method not to override it.
- Use the `jty` library extensively to validate the types that you expect because although the code is written in TypeScript, it may be called with Javascript where typechecking is absent. Also when working with the Small Language Models we cannot be sure that they follow all the instructions. Always verify and fail with meaningful error messages that help you debug the issue and fix it.
- Instead of hard coding values that aren't obvious (e.g. '<|tool_call_start|>'), define const variables with expressive names like `TOOL_CALL_TOKEN` and use that instead.

# Tests

- We use Node.js native testing framework
- Each function is tested in its own `describe()` block
- The first few `test()` blocks test the expected behavior (happy case)
- They are followed by more `test()` blocks which test different edge cases in the order of how common you believe they can be.
- Larget algorithms are implemented in a series of functions that depend on each other. Always start the tests from the leaf nodes of that dependency tree to ensure the logic bottom up.