# Opening prompt for a coding assistant

Copy everything inside the block below into a new chat with any coding assistant. Then attach or point it at `HANDOVER.md` and `README.md`. Never share your `.env` file or API key.

```text
You are joining an existing Python project as my coding partner. Please read HANDOVER.md and README.md in the repository root before doing anything else.

WHAT THIS IS
"EU AI Act & DORA Copilot" is a retrieval-augmented generation (RAG) assistant that answers questions about two EU regulations, the AI Act (Regulation (EU) 2024/1689) and DORA (Regulation (EU) 2022/2554). Every answer must cite the exact Article or Annex it uses, for example [AI Act Art. 27(1)]. The reference implementation runs on Mistral models either through the hosted API (mistral-small-latest, mistral-large-latest, mistral-embed) or fully locally with open-weight models through Ollama. Model clients sit behind one interface, so other providers can be added. It includes an evaluation harness covering retrieval hit@k, MRR, citation accuracy, groundedness, refusals, an LLM judge, latency, tokens and cost.

WHY IT EXISTS
It's a portfolio project for forward-deployed engineering and AI deployment roles. It needs to show three things: RAG built properly for a regulated domain; evaluation discipline, so model choices rest on measurements; and deployment fit, choosing between hosted and fully local to suit the customer's constraints. I'll be walking through it in technical interviews, so the code must stay simple, readable and easy to explain. Please favour clarity over cleverness.

WHERE IT STANDS
- Offline tests pass (python tests/test_pipeline.py: 15 pass, 1 skipped on older Streamlit). CI runs them on Linux and Windows.
- Real EUR-Lex texts ingest correctly: AI Act 113 articles and 13 annexes, DORA 64 articles, 351 chunks in total. A committed snapshot in corpus/ is the fallback.
- eval/verify_golden.py reports 0 problems.
- The CLI and the four-page Streamlit app work end to end.
- The full evaluation (hosted small vs large, BM25 vs hybrid) and sovereign mode (mistral:7b on an 8 GB Mac mini) have been run; results are in the README.
- Environment: macOS, VS Code, Python 3.9 virtual environment. Keep the code 3.9-compatible.

HOW TO WORK WITH ME
1. Before writing code, summarise your understanding of the architecture in 5 to 8 bullet points so I can check it.
2. Follow the "Rules for changes" in HANDOVER.md, especially: keep the offline tests passing and add tests for new behaviour; never print, log or commit the API key; treat the citation label format as a contract; never edit golden.jsonl to inflate scores.
3. Work in small, reviewable steps. For each change, tell me which files you're touching and why, show the diff, and give me the exact terminal commands to run to check it.
4. If something is ambiguous, ask me one clear question rather than guessing.
5. Use British English in docs and UI text.

FIRST TASK
Task 1 from section 8 of HANDOVER.md: help me confirm that automatic fetching from the EU Publications Office's Cellar repository works (python -m regcopilot fetch --refresh -v). Fix fetch.py if a route fails, adding a test. Then help me run the full evaluation (task 2) and fill in the README results table.
```
