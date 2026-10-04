# X post

Two options. The thread is the main one; the single post works if you only
want to make one.

Character counts are the body only — they exclude the trailing link, which X
shortens to 23 characters regardless of length.

---

## Option A — thread (7 posts)

### 1/ (hook)

```
This graph query is valid. It runs. It filters on a real discipline and a real Olympics. It returns 43 real documents with real citations.

It is also completely wrong.

The question asked WHICH event had the most competitors. It answered 43 — how many events there were.

🧵
```

*(~290 chars)*

### 2/

```
Every structural check passes. Valid syntax, real filters, non-empty result, citable sources.

If you only saw the answer, you could not tell. 43 is a perfectly plausible number.

This is why GraphRAG is riskier than plain RAG. RAG usually fails visibly. A subtly wrong graph query doesn't.
```

### 3/

```
So I built the thing that catches it: verify the compiled query against the QUESTION, not just against the schema.

5 deterministic rules, 0 tokens:

• every filter string must appear in the question
• every number too
• constraints must be covered
• the spec must be the shape it claims
• the answer type must match what was asked
```

### 4/

```
Rule 2 is my favourite because it's invisible.

"more than 41 competitors" compiles to min_competitors: 42 about a third of the time.

GSQL filters on `> minCompetitors`, so 42 is off by one. You get 18 where the answer is 20.

No error. Just a quietly wrong count nobody would ever eyeball.
```

### 5/

```
Rejecting isn't enough — each diagnostic says exactly WHAT is wrong, so most failures get repaired by a rule instead of a reasoning loop:

guessed discipline → resolve it from the graph
"immediately before 2016" → walk GAMES_SEQUENCE
"held at <venue> on <date>" → re-issue as a venue+date lookup

That last one: multi-hop 1/9 → 27/28.
```

### 6/

```
100 questions. Same model, same graph, temperature 0:

RAG          0.200  ·  13,817 tokens/correct
GraphRAG     0.370  ·   3,175
Agentic      0.960  ·   1,497

The agent uses FEWER tokens than plain RAG (143k vs 276k) while scoring ~5x higher.

Verification is deterministic. 16/100 cost zero reasoning tokens.
```

### 7/ (close)

```
Two question families still score 0.00 and they're in the report, not hidden — both are schema gaps, not reasoning failures.

And 3 of the worst bugs I found were in the benchmark harness, not the system. Nobody grades your grader.

Built on @TigerGraph Savanna. Code + full results:
github.com/Dhruvpandey1476/Agentic-GraphRAG
```

---

## Option B — single post

```
Built an agentic GraphRAG system on @TigerGraph that verifies its own queries before trusting them.

The failure it catches: a query that's perfectly valid, cites real documents, and answers a different question than the one you asked.

"Which event had the most competitors?" → answered "43" (how many events there were). Every structural check passed.

5 deterministic rules compare the compiled query against the question. When one fails, the diagnostic names the broken slot — so most repairs are rules, not reasoning.

100 questions, same model, temp 0:
RAG 0.200 · GraphRAG 0.370 · Agentic 0.960

And it uses fewer tokens than plain RAG, because verification costs nothing.

github.com/Dhruvpandey1476/Agentic-GraphRAG
```

*(~830 chars — needs X Premium. For the free 280 limit, use post 1 + 6 of the
thread as a 2-post thread instead.)*

---

## Notes before posting

- **Attach a screenshot.** The single highest-value image is the live-query
  page showing all three pipelines disagreeing on the Richmond Olympic Oval
  question — three different medallists, one correct. Second best is the
  agentic trace with `replan_structural` visible.
- **Verify the handle.** `@TigerGraph` is the company account; check whether
  the hackathon asks for a different one, and add the event hashtag if the
  rules require it.
- Post 1 carries the whole thread. If engagement is flat, the hook is the
  thing to change, not the rest.
- Don't claim the 0.960 is held-out. It's a development benchmark — the blog
  says so, and the thread deliberately doesn't imply otherwise.
