## Evaluation — embeddings `qwen3-embedding:0.6b`

### Retrieval (26 natural questions, 10 keyword queries)

| Mode | Natural Hit@1 | Natural Hit@4 | Keyword Hit@1 | Keyword Hit@4 | MRR@10 (all) |
|---|---|---|---|---|---|
| hybrid | 92% | 100% | 100% | 100% | 0.965 |
| dense | 100% | 100% | 80% | 100% | 0.968 |
| bm25 | 69% | 88% | 100% | 100% | 0.836 |

### End-to-end (`gemma3:4b`, hybrid, k=4)

| Question type | n | Correct answer | Cites a relevant chunk |
|---|---|---|---|
| single-turn | 36 | 100% | 94% |
| follow-up | 4 | 100% | 100% |
| out-of-scope | 4 | 100% | – |

Follow-up retrieval Hit@1: raw question 2/4 → condensed question 4/4
Mean latency per question: 2.7s

#### Failures
- **When is the hotline open on Saturdays?** (correct=True, cited relevant=False)  
  'We offer support on Saturdays from 9:00 - 17:00 [2].'
- **ombudsman** (correct=True, cited relevant=False)  
  'The Ombudsman process is available for unresolved disputes [4].'
