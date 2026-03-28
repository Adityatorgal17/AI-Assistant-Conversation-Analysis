# Conversation Insights

This package is the end-to-end analysis pipeline for the HelioAI assignment.

Its job is to take raw ecommerce assistant conversations and turn them into:

- cleaned conversation transcripts
- structured conversation features
- LLM-reviewed outcome labels
- dashboard-friendly conversation rows
- per-brand insight summaries
- one global summary across all brands

The goal is not just to "process data", but to surface useful insight about where the assistant succeeds, where it fails, and why.

## What A Reviewer Should Know First

This project uses a hybrid approach:

- deterministic ETL and heuristics for things we can observe directly
- LLM review only for meaning-heavy judgments

That design choice is intentional.

We do not ask the LLM to infer everything from scratch. Instead:

- the ETL layer reconstructs each conversation accurately
- the feature layer extracts stable behavioral signals
- the LLM adds semantic judgments like resolution, recommendation quality, safety risk, and concise explanations
- a final scoring layer converts those signals into `good`, `neutral`, and `bad` quality buckets

This keeps the pipeline easier to debug, cheaper to run, and more reliable than an LLM-only design.

## Pipeline Overview

The pipeline runs in this order:

1. Load raw `conversations.json` and `messages.json`
2. Rebuild each conversation as a timestamp-sorted transcript
3. Clean assistant text by removing appended payload noise such as `End of stream` JSON
4. Extract deterministic metadata like message counts, link clicks, product views, login clicks, and observed product recommendations
5. Send each conversation to the LLM review step for semantic interpretation
6. Apply normalization and post-processing rules to the LLM output
7. Compute a final score and quality label
8. Materialize JSON outputs for dashboards and summaries
9. Optionally write processed collections to MongoDB

In code, the main orchestration happens in:

[`main.py`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/main.py)

## File-By-File Guide

### Core Pipeline Files

- [`main.py`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/main.py)
  Runs the full pipeline, writes output JSON files, and optionally writes MongoDB collections.

- [`etl.py`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/etl.py)
  Reconstructs grouped conversations from raw conversation and message dumps.

- [`features.py`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/features.py)
  Extracts deterministic conversation-level features such as event counts, product interactions, login clicks, and structure metrics.

- [`llm_review.py`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/llm_review.py)
  Builds the prompt, calls Groq, caches LLM results, normalizes labels, and computes the final quality score.

- [`insights.py`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/insights.py)
  Converts feature records into dashboard rows, per-widget summaries, and a global summary.

- [`dashboard.py`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/dashboard.py)
  Streamlit app for reviewing outputs at global, widget, and conversation level.

### Support Files

- [`text_utils.py`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/text_utils.py)
  Text cleaning, URL parsing, lightweight language detection, product-link detection, and brand-name inference helpers.

- [`models.py`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/models.py)
  Dataclasses for grouped records, feature records, dashboard rows, and summary records.

- [`config.py`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/config.py)
  Shared paths and Mongo settings.

- [`env_utils.py`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/env_utils.py)
  Small `.env` loader for Groq API keys.

- [`mongo_store.py`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/mongo_store.py)
  Writes processed outputs into MongoDB collections.

## Raw Input Schema

The project expects the assignment data in the repo root:

- `conversations.json`
- `messages.json`

The current dataset contains:

- `300` conversations
- `1525` messages
- `3` widget IDs

Important characteristics of the raw data:

- one conversation contains many messages
- `messageType="text"` is the actual chat
- `messageType="event"` captures actions like product views, quick actions, and link clicks
- many agent messages contain appended structured payloads that must be cleaned before analysis

## Stage 1: Grouped Conversation ETL

The grouped conversation record is the pipeline's source of truth for each conversation.

Produced in:

[`etl.py`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/etl.py)

### What happens here

- load raw JSON arrays
- map messages to their parent conversation
- sort messages by timestamp
- preserve both `rawText` and `cleanText`
- infer `brandName` from the dominant domain seen in outgoing links

### Why this matters

This step makes later debugging easy.

For any conversation ID, a reviewer can trace:

`raw messages -> grouped transcript -> feature record -> dashboard row`

That traceability was a deliberate design choice.

## Stage 2: Deterministic Feature Extraction

Produced in:

[`features.py`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/features.py)

### What is extracted deterministically

#### Structure

- `num_messages`
- `num_event_messages`
- `num_user_turns`
- `num_agent_turns`
- `first_user_text`
- `who_ended_conversation`

#### Observable interaction signals

- `linkClickCount`
- `productClickCount`
- `feedbackClickCount`
- `productViewCount`
- `loginClickCount`
- `hasWhatsAppHandoff`
- `containsProductLinks`
- `numRecommendationsObserved`
- `recommendedProductNames`

### Why these are deterministic

These signals are directly observable from text or event metadata.

Examples:

- if a user clicked a login link 24 times, we do not need an LLM to notice that
- if the assistant includes a real product link, we can record that before the LLM sees the transcript

## Stage 3: LLM Review

Produced in:

[`llm_review.py`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/llm_review.py)

This is the semantic judgment layer.

### What the LLM decides

- `initialIntent`
- `languageStyle`
- `hasSafetySensitiveContext`
- `userRepetition`
- `frustrated`
- `assistantRepetition`
- `assistantShortAnswer`
- `assistantEvasiveAnswer`
- `containsOrderInstructions`
- `containsSafetyDisclaimer`
- `containsPossibleClaimRisk`
- `recommendationGiven`
- `recommendationRelevant`
- `badRecommendation`
- `success`
- `dropOff`
- `unresolved`
- `primaryProblem`
- `conversationOutcome`
- `issues`
- `summary`
- `feedbackText`

### Prompt design choices

The prompt is intentionally structured around:

- transcript-first reasoning
- metadata as supporting evidence only
- conservative labeling when uncertain
- explicit examples for intent and language
- rules for login loops, order instructions, repetition, and distinguishing real recommendations from navigation links

### Cache behavior

Reviews are cached in:

[`outputs/llm_review_cache.json`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/outputs/llm_review_cache.json)

The cache uses `__schemaVersion`.

This is important because:

- if the prompt changes materially, the cache schema should also change
- otherwise old reviews may silently remain in place and hide prompt improvements

## Stage 4: Post-Processing And Normalization

Also handled in:

[`llm_review.py`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/llm_review.py)

The raw LLM answer is not used blindly.

We apply final rules such as:

- infer intent if the model returns `other`
- infer language if the model leaves it unknown
- infer recommendation flags from deterministic product signals when appropriate
- infer `primaryProblem` when the transcript clearly shows `login_loop`, `order_friction`, `bad_recommendation`, or `risk_flagged`
- normalize `conversationOutcome` to match stronger failure modes

This post-processing keeps outputs more stable and makes the system less brittle than a pure prompt-only approach.

## Stage 5: Derived Score And Quality Bucket

The final score and `quality` label are computed in:

[`llm_review.py`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/llm_review.py)

### Scoring logic

Positive points:

- `+2` if there was link-click engagement and the problem is not a login loop
- `+2` if the user interacted with products
- `+2` if a relevant recommendation led to a product-view event

Negative points:

- `-1` for drop-off
- `-2` for feedback-click friction
- `-1` for assistant repetition
- `-1` for bad recommendation
- `-2` for frustration
- `-2` for unresolved status
- `-1` for `login_loop`, `order_friction`, or `unresolved_need`

### How `quality` is assigned

- `score >= 2` -> `good`
- `score >= 0 and < 2` -> `neutral`
- `score < 0` -> `bad`

### Interpretation

`good` does not mean "perfect conversation".

It means the observable behavior and semantic judgment together suggest a healthy outcome.

Likewise, `bad` does not mean the assistant was always wrong.

It means the conversation contains enough friction, unresolved need, repetition, or failed flow signals to deserve attention.

## How The Output Summaries Are Built

The summaries are generated in:

[`insights.py`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/insights.py)

### `dashboard_rows.json`

This is the flattened, reviewer-friendly row for each conversation.

Each row contains:

- the conversation ID and widget ID
- intent
- quality and score
- success / unresolved / drop-off flags
- recommendation and click signals
- top problem and outcome
- one reviewer-readable explanation in `feedbackText`

### `widget_insights.json`

One summary per widget/brand.

#### `qualityBreakdown`

Count of how many conversations for that widget ended up as:

- `good`
- `neutral`
- `bad`

This comes directly from the final derived quality score.

#### `outcomeBreakdown`

Count of conversations by semantic outcome label:

- `resolved`
- `unresolved`
- `drop_off`
- `frustrated`
- `order_friction`
- `login_loop`
- `bad_recommendation`
- `risk_flagged`
- `neutral`

This comes from the normalized `llmReview.conversationOutcome`.

#### `topIntents`

Most common values of:

- `llmReview.initialIntent`

This shows what kinds of conversations dominate a widget.

#### `topProblems`

Most common non-null values of:

- `llmReview.primaryProblem`

This shows recurring failure modes such as:

- `unresolved_need`
- `order_friction`
- `login_loop`
- `risk_flagged`

#### `summaryPoints`

Human-readable highlights generated from counts such as:

- how many conversations are `bad`
- the top intent
- unresolved count
- login-loop count
- order-friction count
- whether recommendation flows lead to follow-through events

### `global_summary.json`

This is the same idea as `widget_insights.json`, but aggregated across the full dataset.

It contains:

- `totalConversations`
- `qualityBreakdown`
- `outcomeBreakdown`
- `topIntents`
- `topProblems`
- `summaryPoints`

## Generated Output Files

Outputs are written into:

[`outputs`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/outputs)

Main files:

- `grouped_conversations.json`
- `conversation_features.json`
- `dashboard_rows.json`
- `widget_insights.json`
- `global_summary.json`
- `llm_review_cache.json`

### What each file is for

- `grouped_conversations.json`
  Best file for debugging transcript reconstruction and cleaning.

- `conversation_features.json`
  Best file for understanding the full structured feature set per conversation.

- `dashboard_rows.json`
  Best file for a reviewer who wants a compact, one-row-per-conversation view.

- `widget_insights.json`
  Best file for understanding each brand/widget at a glance.

- `global_summary.json`
  Best file for high-level overall conclusions.

- `llm_review_cache.json`
  Best file for inspecting raw LLM review outputs before final derived scoring.

## How To Read One Conversation End-To-End

For a given `conversationId`, the recommended review flow is:

1. Find it in `messages.json`
2. Confirm the grouped transcript in `grouped_conversations.json`
3. Inspect the structured features in `conversation_features.json`
4. Check the flattened row in `dashboard_rows.json`
5. Inspect the raw cached LLM output in `llm_review_cache.json` if needed

This makes the system audit-friendly.

## Why These Decisions Were Made

Key design decisions:

- keep raw reconstruction separate from derived features
- make conversation-level outputs easy to trace
- avoid overusing the LLM for directly observable facts
- use LLM review for meaning-heavy judgments
- keep summary outputs reviewer-friendly
- store enough intermediate structure to debug mistakes

## Known Limitations

Current limitations to be aware of:

- some labels still depend on prompt quality
- quality scoring is rule-based, so it is interpretable but not probabilistic
- new datasets with very different URL structures or event naming may require small heuristic updates
- the LLM cache must be refreshed when the prompt changes materially

## Setup

### 1. Clone The Repository

```bash
git clone https://github.com/Adityatorgal17/AI-Assistant-Conversation-Analysis
cd AI-Assistant-Conversation-Analysis
```

### 2. Create And Activate A Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r conversation_insights/requirements.txt
```

### 3. MongoDB Setup

If you want processed outputs written to MongoDB, make sure MongoDB is running on:

```text
mongodb://localhost:27017/
```

Database used by default:

```text
helio_intern
```

### 4. Groq API Keys

Create Groq keys and place them in:

[`conversation_insights/.env`](/home/aditya17/Aditya/summer/Assignment/HelioAI/conversation_insights/.env)

Example:

```env
GROQ_API_KEY1=your_key_here
GROQ_API_KEY2=your_key_here
GROQ_API_KEY3=your_key_here
```

Multiple keys help with rate limits.

Practical note:

- in our runs, one API key typically started hitting rate limits after roughly `60-70` conversation reviews
- for the current dataset of `300` conversations, we used `5` Groq API keys
- for larger datasets, make sure you provision enough API keys to cover the dataset size comfortably

## Run Commands

### Run The Pipeline

```bash
python3 -m conversation_insights --output-dir conversation_insights/outputs
```

### Run And Write To MongoDB

```bash
python3 -m conversation_insights --output-dir conversation_insights/outputs --write-mongo --mongo-uri mongodb://localhost:27017/
```

### Force Fresh LLM Reviews

```bash
python3 -m conversation_insights --output-dir conversation_insights/outputs --clear-llm-cache
```

### Test With A Different Dataset

If you want to test the same pipeline on a different dataset:

1. Replace the repo-root `conversations.json` and `messages.json` files with the new dataset files.
2. Run:

```bash
python3 -m conversation_insights --output-dir conversation_insights/outputs --clear-llm-cache
```

This will rebuild grouped conversations, features, summaries, and fresh LLM review outputs for that dataset.

## Dashboard

To view the results directly, run the Streamlit app:

```bash
streamlit run conversation_insights/dashboard.py
```

Then open:

```text
http://localhost:8501
```

This will let you inspect the generated insights conversation-by-conversation and widget-by-widget.

## Final Reviewer Takeaway

This package is designed to be:

- understandable
- traceable
- debuggable
- reviewer-friendly

The reviewer should be able to answer all of these by reading the code and outputs:

- What is the pipeline doing at each stage?
- Which decisions are heuristic vs LLM-driven?
- Why is a conversation marked `good`, `neutral`, or `bad`?
- Why does a widget have a certain `qualityBreakdown`, `topIntents`, or `topProblems`?
- Can one conversation be traced from raw input to final summary?

That transparency is a core part of the implementation, not an afterthought.
