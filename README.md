# mtn_agents

`mtn_agents` is a Python library for building AI agent workflows around indexed project knowledge.

The library was built as the AI layer for a larger product. It provides reusable components for preparing files, indexing their contents, retrieving relevant context, and running language-model inference over that context.

## What It Does

At a high level, `mtn_agents` helps with:

- Reading and preparing reference files for indexing.
- Uploading document content into a retriever or vector index.
- Searching indexed knowledge during a conversation or query.
- Running LLM-based inference with retrieved context.
- Building agent-style flows for question answering, support, sales, teaching, and presentation-related use cases.
- Generating higher-level outputs such as summaries, suggested questions, conversation insights, and presentation feedback.

## Purpose

This repository is intended to be used as a library inside larger AI applications. It is not a standalone product by itself. The main value is in the reusable indexing, retrieval, and inference utilities that can be composed into different agent experiences.

## Core Idea

The general workflow is:

1. Provide files, URLs, or reference text.
2. Process and index that content.
3. Retrieve the most relevant indexed context for a user query.
4. Send the query and retrieved context to an LLM.
5. Return an answer or structured result for the application.

## Package

The import package is:

```python
import mtn_agents
```

