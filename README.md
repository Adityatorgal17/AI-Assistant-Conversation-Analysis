# Conversation Insights Pipeline

ELT + LLM pipeline for analyzing ecommerce assistant conversations and producing useful insights and displaying on dashboard.

## What This Project Does

Input:

- `conversations.json`
- `messages.json`

Output:

- `conversation_insights/outputs/grouped_conversations.json`
- `conversation_insights/outputs/conversation_features.json`
- `conversation_insights/outputs/dashboard_rows.json`
- `conversation_insights/outputs/widget_insights.json`
- `conversation_insights/outputs/global_summary.json`
- `conversation_insights/outputs/llm_review_cache.json`

## Pipeline Flow

`input (conversations, messages)` -> `grouped conversations` -> `deterministic features` -> `LLM extracted features` -> `deterministic insights` -> `LLM generated insights` -> `output files`

This is implemented as:

1. Build grouped conversations from raw JSON.
2. Extract deterministic conversation features.
3. Run mandatory LLM review (`run_llm_reviews`) to enrich semantic labels.
4. Build deterministic base insights.
5. Enhance insights with LLM root causes/actions and guarded discovery.
6. Write final JSON outputs for dashboard/analysis.

## Pipeline Stages

1. `etl.py`

- Reconstruct conversation transcripts and clean agent text.

2. `features.py`

- Compute deterministic conversation features.

3. `llm_review.py`

- Run `run_llm_reviews` for conversation-level LLM review + cache.

4. `insights_generator.py`

- Generate base widget/global recommendations.

5. `llm_insights_generator.py`

- Enrich deterministic recommendations with LLM-generated root causes/actions and discovery guardrails.

6. `insights.py`

- Build final dashboard rows, widget summaries, and global summary.

7. `dashboard.py`

- Streamlit UI for global and widget-level analysis.

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r conversation_insights/requirements.txt
```

### 3. Configure `.env`

Create `conversation_insights/.env`:

```dotenv
# Single key
GROQ_API_KEY=gsk_xxx

# OR multiple keys (recommended for fallback / rate-limit handling)
GROQ_API_KEY1=gsk_xxx
GROQ_API_KEY2=gsk_xxx
GROQ_API_KEY3=gsk_xxx
```

Notes:

- Multiple keys are auto-discovered and rotated when a key hits 429/rate limits.

## Run Pipeline

```bash
python3 -m conversation_insights --output-dir conversation_insights/outputs
```

Useful flags:

```bash
# Force fresh LLM review (ignore existing review cache)
python3 -m conversation_insights --output-dir conversation_insights/outputs --clear-llm-cache

# Optional Mongo write
python3 -m conversation_insights --output-dir conversation_insights/outputs --write-mongo --mongo-uri "mongodb://localhost:27017/"
```

## View Dashboard

```bash
streamlit run conversation_insights/dashboard.py
```

In dashboard:

- **Global Summary** view: full global metrics and summary on one page.
- **Widget Insights** view: pick a widget from dropdown and view:
  - widget recommendations
  - conversation table for that widget
  - conversation detail/transcript for that widget

## Project Structure (Key Files)

- `conversation_insights/main.py` - pipeline orchestrator
- `conversation_insights/etl.py` - extract/load grouped transcripts
- `conversation_insights/features.py` - deterministic transforms
- `conversation_insights/llm_review.py` - conversation-level LLM review + cache (`run_llm_reviews`)
- `conversation_insights/insights_generator.py` - base recommendations
- `conversation_insights/llm_insights_generator.py` - LLM enhancement of recommendations
- `conversation_insights/insights.py` - final summary builders
- `conversation_insights/dashboard.py` - Streamlit app

## Quick Troubleshooting

- **No API key found**

  - Verify `conversation_insights/.env` exists and key names are valid.
- **429 rate limit / token limit**

  - Add more `GROQ_API_KEY*` entries.
  - Re-run pipeline; key rotation handles fallback automatically.
- **Dashboard shows stale data**

  - Re-run pipeline to regenerate outputs.
  - Refresh Streamlit page.
