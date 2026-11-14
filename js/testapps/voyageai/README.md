# Voyage AI test app

Exercises the `@genkit-ai/voyageai` plugin: text, contextualized and multimodal
embedders and rerankers. Set `VOYAGE_API_KEY`, then:

```bash
cd js/testapps/voyageai
genkit start -- tsx --watch src/index.ts
```

Flows: `embedText`, `embedContextualized`, `embedMultimodal`, `rerank`. Each
takes the model name explicitly; the plugin has no default model.
