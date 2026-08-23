\## Architecture



InsightFlow AI follows a grounded and auditable analytics workflow:



1\. The user uploads a supported data or document file.

2\. The unified loader extracts a tabular DataFrame.

3\. Searchable PDFs are parsed directly.

4\. Scanned PDFs are processed with English and Myanmar OCR.

5\. A deterministic parser or local Qwen3 planner interprets the question.

6\. The generated analysis plan is validated with Pydantic.

7\. Metrics, dimensions, and filters are grounded to the current DataFrame.

8\. Untrusted model tool instructions are replaced with approved deterministic steps.

9\. Revenue and profit quality gates validate the current DataFrame.

10\. Pandas performs the actual calculation.

11\. Streamlit displays the result, visualization, and evidence.



Processing flow:



User Question  

→ Rule Parser or Local Qwen Planner  

→ Structured AIAnalysisPlan  

→ Pydantic Validation  

→ DataFrame Schema Grounding  

→ Approved Tool-Step Reconstruction  

→ Financial Quality Gate  

→ Deterministic Pandas Execution  

→ Verified Chart and Report





\## Trust Boundary



The local Qwen model interprets analytical questions and produces structured

analysis plans.



The local model does not calculate revenue, profit, totals, rankings, filtered

results, or correlations. These operations are performed by deterministic

Pandas code after schema grounding and quality-gate validation.



Model-generated tool instructions are treated as untrusted. InsightFlow AI

reconstructs an approved sequence of deterministic operations before execution.





\## Verified Results



The regression dataset produces the following validated results:



\- Total revenue: 6,355

\- Total profit: 1,335

\- Yangon revenue: 4,415

\- Yangon filtered rows: 5

\- Highest-revenue product: Laptop

\- Laptop revenue: 4,250

\- Full local automated suite: 16 passed

\- CI-equivalent deterministic suite: 12 passed, 4 Ollama tests deselected





\## Local LLM Setup



InsightFlow AI uses Ollama with Qwen3 4B for optional local analysis planning.



Install or start Ollama, then pull the model:







