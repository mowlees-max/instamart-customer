# Instamart Customer Intelligence & Support Copilot

## Overview

This repository implements the four-part synthetic Instamart Customer Intelligence & Support Copilot capstone. It cleans deterministic fictional customer, product, order and ticket exports; stores trusted tables in CSV and SQLite; segments customers and predicts lapse risk; redacts and routes support tickets; and provides policy retrieval plus a bounded, database-backed refund agent. All amounts are in INR. The supplied `generate_data.py` and `mock_llm.py` are included verbatim from the brief; every model call is offline and deterministic.

## Reproduce

Run these commands from the repository root. Python 3.11+ and SQLite 3.25+ are required.

1. `python -m venv .venv`
2. Windows PowerShell: `.venv\\Scripts\\Activate.ps1` (macOS/Linux: `source .venv/bin/activate`)
3. `python -m pip install --upgrade pip`
4. `python -m pip install -r requirements.txt`
5. `python generate_data.py`
6. `python run_project.py`
7. `python complete_project.py`

The generator's final line is `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. `run_project.py` audits, cleans, validates, trains the Part B models, runs few-shot triage and evaluates the policy RAG. `complete_project.py` produces the zero-shot/budget comparison, all seven refund-agent runs, stop-safe demonstrations, crash/resume checkpoint and evaluation summaries. Generated CSV, SQLite, JSONL and checkpoint outputs are written under `data/`, `outputs/` and `state/`.

## Part A

### Raw audit

The raw tables have these shapes and pandas dtypes when loaded with `keep_default_na=False`:

| Table | Shape | Dtypes |
|---|---:|---|
| customers | 1,200 × 5 | customer_id object; city object; signup_date object; age object; gender object |
| products | 40 × 4 | product_id object; product_name object; category object; price_inr int64 |
| orders | 13,084 × 9 | order_id object; customer_id object; product_id object; quantity int64; order_ts object; status object; discount_pct int64; order_amount object; delivery_minutes object |

With `keep_default_na=False`, the CSV parser reports zero `NaN` values in every column; blank strings are separately audited rather than silently converted to NaN. Products have no blank fields. The order audit is: 13,084 raw rows; 119 exact duplicate rows; 656 pending rows; 12,309 distinct order IDs; 120 non-numeric amounts; 106 negative delivery times; 2,541 blank delivery times. Customer audit: 61 blank ages; 6 ages outside 10–100; 24 raw city spellings; 10 distinct non-blank gender spellings; 30 blank genders.

### Cleaning and storage

- Customer city: strip surrounding whitespace and title-case. Final counts: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184.
- Gender: normalize case and map `male`/`m` to `male`, `female`/`f` to `female`, and blanks or other values to `unknown`. Counts: female 610, male 560, unknown 30.
- Age: parse numeric; values outside 10–100 become missing; fill missing values with the median age of that customer's city using `groupby(...).transform(...)`. Final missing ages: 0; all ages are 18–60. Signup date is parsed and stored as ISO date text.
- Orders: parse `order_ts`; remove 119 exact duplicates; sort by order timestamp and retain the latest row for each `order_id` (resolving 656 earlier pending records); coerce 120 bad amounts to missing and impute from catalogue price × quantity × (1 − discount/100); set 106 negative delivery durations to missing; impute completed-order delivery durations with the completed-order median for the product category. Cancelled delivery durations remain missing.
- Final orders: 12,309 rows, 11,110 completed and 1,199 cancelled; no pending rows; no missing amount; no missing completed delivery duration; 1,199 cancelled delivery durations missing. Completed revenue is INR 2,998,668.80.

The validation script asserts the city/gender/age constraints, unique order IDs, absence of pending rows, complete amounts, completed/cancelled delivery invariants and SQLite table presence. It prints `ALL CHECKPOINTS PASSED`.

### SQL results

Seven executable queries are in `sql/queries.sql`; each is run against `data/instamart.db` using `pandas.read_sql_query`.

| Query | Result |
|---|---|
| Q1 WHERE then HAVING | 30 customers |
| Q2 city × category revenue join | 30 rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics, INR 233,909.45 |
| Q3 products above average catalogue price | 11 products |
| Q4 customers with no completed order | 25 customers |
| Q5 above-average completed spend via two CTEs | 550 customers; mean customer spend INR 2,552.06 |
| Q6 city-partitioned RANK, DENSE_RANK, ROW_NUMBER | Top: Bengaluru C0082 INR 12,856.20; Chennai C0930 INR 10,151.40; Delhi C0291 INR 13,742.60; Hyderabad C0743 INR 12,730.05; Mumbai C0693 INR 14,766.55; Pune C0426 INR 10,377.10. Eight rows have RANK different from ROW_NUMBER. |
| Q7 daily running revenue | 183 daily rows; final cumulative revenue INR 2,998,668.80 |

Tied spend values share a rank under `RANK()` (and the next rank is skipped), while `ROW_NUMBER()` assigns a distinct sequential number to every row. `DENSE_RANK()` shares ranks without leaving gaps.

### pandas cross-check and exploration

The Q2 pivot table grand total equals the SQL total, INR 2,998,668.80. Completed-order city summaries include Bengaluru: sum INR 523,643.40, mean INR 276.47, count 1,894; Mumbai: sum INR 479,058.50, mean INR 297.00, count 1,613. Across completed orders, mean amount is INR 269.91, median INR 135.00 and skewness 4.61; Q1 is INR 64.80 and Q3 is INR 297.00. The 1.5×IQR fences are INR −283.50 and INR 645.30, with zero observations below and 897 above. The data describes synthetic completed grocery, personal-care and electronics orders. Spend is strongly right-skewed: most orders are modest, while a small number of expensive baskets form the upper tail. The upper-tail observations are statistical outliers, not automatically errors; impossible negative delivery durations and malformed raw amounts were data-quality anomalies handled during cleaning. The largest absolute correlation is order amount versus discount percentage (−0.095); order amount versus quantity is +0.087. These are weak associations, and a correlation of this size does not establish causation.

## Part B

### RFM and segmentation

RFM uses completed orders through 2025-09-30, with snapshot 2025-10-01. There are 1,175 customers: recency median 19 days (min 0, max 180), frequency mean 9.46 (max 33), and monetary mean INR 2,552.06 (max INR 14,766.55).

K-means diagnostics on StandardScaler-transformed RFM:

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.481 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 is selected: it has the strongest silhouette in this run and yields actionable groups. Inertia always decreases as K increases because adding centroids cannot increase the minimum within-cluster squared distance; K=8 therefore fits at least as tightly as K=2 by construction. That monotonic decrease does not measure whether the extra segments are stable or useful, so it cannot select K by itself.

| Segment ID | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Name / action |
|---:|---:|---:|---:|---:|---|
| 0 | 436 | 28.86 | 3.77 | 1,057.62 | Occasional Shoppers — relevant replenishment reminders |
| 1 | 138 | 24.12 | 6.77 | 7,129.70 | Premium — proactive service and high-value retention |
| 2 | 419 | 8.77 | 19.40 | 3,410.02 | Frequent Essentials — subscriptions and basket convenience |
| 3 | 182 | 109.04 | 2.20 | 685.99 | Dormant / At Risk — win-back offer and feedback request |

Segment 1 is Premium because it has the highest mean monetary value. Cluster IDs are arbitrary labels, not an ordinal scale. PCA on scaled RFM explains 0.634 and 0.218 of variance (total 0.852).

### Lapse model

The pre-cutoff cohort contains 1,156 customers; 443 lapsed (38.32%). Split: 924 train and 232 test. Baseline always predicts not-lapsed: accuracy 61.6%, lapsed-class F1 0. Train class counts are 570 not-lapsed and 354 lapsed. The full-cohort frequency mean is 8.03; the training-fitted scaler's frequency mean is 8.11. They differ because the scaler is fitted only on the training partition, not the full cohort.

Features use only orders before 2025-09-01 plus clean customer data: recency, frequency, monetary, mean discount, cancellation rate, mean delivery, favourite category, city, age, gender and tenure. September orders are used only for the target label. Segment labels are not model features. Numeric imputation, scaling and one-hot encoding are fitted inside the training pipeline; no encoder, scaler or resampler sees test rows, and the test set is used only for final evaluation. No feature column contains missing values.

| Model | Best parameters | Best CV F1 | Test confusion matrix (TN, FP / FN, TP) | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---|---:|---|---:|---:|---:|
| Logistic regression | C=100 | 0.651 | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | depth=8, min_samples_leaf=10 | 0.687 | [[110, 33], [32, 57]] | 0.633 | 0.640 | 0.637 |

The persisted model is the random forest because it had the higher cross-validated F1 (0.687); its test lapsed-class F1 is 0.637, above the 0.55 acceptance threshold. Missing a lapsing customer means losing a chance to intervene and potentially losing future orders; wrongly targeting a loyal customer wastes incentive spend and can annoy them. The selected model balances these costs, while the recall/precision trade-off should be monitored with campaign capacity and incentive cost.

Training-only oversampling duplicates lapsed training examples to balance the classes at 570 / 570. On the unchanged test set, balanced logistic regression has precision 0.600, recall 0.809 and F1 0.689 (confusion matrix [[95, 48], [17, 72]]), compared with unbalanced logistic precision 0.644, recall 0.652 and F1 0.648. Recall moves most: the retention team catches more lapsed customers, at the cost of more false-positive outreach.

Top random-forest features: monetary (0.280), frequency (0.278), recency (0.109), average delivery (0.060), cancellation rate (0.052). Logistic coefficients: frequency −1.697 (higher frequency lowers predicted lapse odds); favourite category Electronics −1.135 and Grocery −0.449 (relative to the dropped reference category, these reduce predicted lapse odds in this fitted model). The saved `outputs/lapse_model.joblib` is reloaded and reproduces test predictions exactly. The top-10 Premium test retention list contains 1 actually lapsed customer. The retention list is illustrative, not a leakage-free evaluation: its Premium flag is a post-hoc whole-window label that includes September.

## Part C

### Guardrails, privacy and classification

The pipeline processes the 164 tickets in file order. Four empty tickets are `skipped_empty`; four long tickets are `flagged_long` and are never sent to the model. Their estimated input length is 734 tokens, above the 400-token limit. Redaction counts across the eligible text: 55 emails, 91 phone numbers and 132 order IDs (278 substitutions total); assertions confirm no matching PII remains after redaction. The model's user message contains only the redacted ticket text.

Both prompts require JSON-only output with one of billing, delivery, technical or fraud. The zero-shot prompt has no `EXAMPLES:` block. The few-shot prompt has labelled fraud, delivery and technical examples. Each call retries transient failures up to three times with exponential backoff and jitter; the full run encountered 19 transient failures and recovered all 19. The retry log records ticket, retry number and wait.

| Prompt | Classified | Accuracy | Total tokens | Tokens / classified ticket |
|---|---:|---:|---:|---:|
| Zero-shot | 147 | 85.7% (126/147) | 7,372 | 50.15 |
| Few-shot | 147 | 100% (147/147) | 12,520 | 85.17 |

Few-shot costs 35.02 additional tokens per classified ticket (5,148 additional tokens overall) and improves accuracy by 14.3 percentage points in this deterministic mock. That extra cost is worthwhile when misrouting fraud or support cases is more expensive than the additional inference budget; a low-risk high-volume queue may prefer the cheaper zero-shot prompt with human review for uncertainty. Zero-shot confusion counts: billing→billing 38, fraud→billing 11, technical→billing 10, delivery→delivery 36, fraud→fraud 23, technical→technical 29 (all other cells 0).

Few-shot status counts: 135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output`, 4 `flagged_long`, 4 `skipped_empty`. Keyword recovery uses `re.findall(r"[a-z]+", text.lower())`; valid JSON outside the allow-list is not guessed and goes to human review. A zero-shot 6,000-token budget demo exhausted the budget on 29 tickets. Final few-shot triage uses a 100,000-token budget and writes 164 rows to `outputs/triage_results.csv`. Human review receives skipped-empty, flagged-long, invalid-output and budget-exhausted cases; fraud goes to high-priority security; Premium customers otherwise go to high-priority premium desk; remaining tickets go to the normal support queue. Rule-validated categories are the JSON/keyword classified results; uncertain or guardrail-blocked tickets go to humans.

## Part D

### Policy RAG

Five policy documents are split into 40-word chunks with 10-word overlap, producing 15 chunks: refund 4, delivery 3, escalation 3, payment 3 and privacy 2. `TfidfVectorizer` represents the chunks and cosine similarity ranks them. Term frequency measures how often a term occurs in a chunk; inverse document frequency downweights terms common across chunks. TF-IDF is lexical, so equivalent wording with different vocabulary can be missed.

| Chunking | Answerable questions with expected doc in top 3 | Expected fact present | Unanswerable fallback |
|---|---:|---:|---:|
| 10-word overlap | 12/12 | 12/12 | 2/2 |
| Zero overlap | 12/12 | 10/12 | 2/2 |

With zero overlap, the delivery-time refund question and the failed-payment return-time question lose the expected fact because the relevant policy sentence is split across chunk boundaries. The assistant returns the model's fallback when retrieved context does not contain an answer.

### Refund agent

The five tools are `get_order`, `search_policy`, `calculate_refund`, `issue_refund` and `escalate_to_human`. The agent checks the SQLite order, retrieves policy context, calculates a full refund only for completed orders delivered in more than 45 minutes, and escalates amounts above INR 1,000 rather than issuing them automatically. Tool calls and observations are appended to the ReAct message history. Ledger writes are local JSONL simulation records, not real payments.

| Scenario | Status | Refund amount (INR) | Model calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 897 |
| S2 | REFUND | 135.00 | 5 | 897 |
| S3 | ESCALATE | 1,999.00 | 5 | 884 |
| S4 | DENIED | 0.00 | 4 | 605 |
| S5 | DENIED | 0.00 | 3 | 337 |
| S6 | NOT_FOUND | 0.00 | 2 | 72 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,000 |

Stop-safely demonstrations: S1 with token limit 200 ends `FALLBACK_TOKEN_LIMIT` (3 model calls, 354 tokens); S1 with time limit 0 ends `FALLBACK_TIMEOUT` before any model call (0 calls, 0 tokens). S7 reaches `FALLBACK_MAX_ITERATIONS` at the default six-call limit. Crash-and-resume test: after the second tool observation checkpoint, the run resumes to REFUND INR 128.25. Tools executed across crash and resume, in order: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`; no tool is repeated.

ReAct makes one model call per decision and uses the latest tool observation to choose the next action; S1 takes five calls, S5 three and S6 two. A plan-and-execute agent could request the full plan in one model call and replan only after an error, reducing calls and usually tokens for predictable paths. However, a refund workflow must react to observations such as NOT_FOUND, cancelled status or an amount requiring escalation; blindly executing a stale plan risks an unsafe action. ReAct is preferred here because the extra calls buy step-by-step policy and database verification, with explicit iteration, token and time limits plus human fallback.

## Limitations and Next Steps

All records and policies are synthetic and fictional. `mock_llm.py` is a deterministic offline simulator, not a production language model; TF-IDF is lexical and may miss semantic paraphrases. The refund ledger records simulated actions only and does not connect to a payment processor. The Premium retention list is illustrative because Premium is assigned using whole-window RFM that includes September; it is not a leakage-free evaluation. Next steps would include access controls, real policy governance, secure PII handling, monitoring, drift checks, calibrated thresholds, human approval workflows and integration tests against a sandbox payment service.
