# Power BI Setup

## 1. Generate the workbook

Publish at least one validated run from Streamlit, or call
`publish_to_powerbi_workbook()` directly. The default path is:

```text
powerbi_output/insightflow_powerbi.xlsx
```

## 2. Connect Power BI Desktop

1. Open Power BI Desktop.
2. Select **Home > Get data > Excel workbook**.
3. Select `powerbi_output/insightflow_powerbi.xlsx`.
4. Load all five tables:
   - AnalysisRuns
   - AnalysisResults
   - AIResponses
   - QualityEvidence
   - SourceFiles

## 3. Create relationships

Create one-to-many relationships from `AnalysisRuns[run_id]` to the `run_id`
field of the other four tables. Use single-direction filtering from
AnalysisRuns to each child table.

## 4. Recommended report pages

### Executive Overview

- Latest question
- Latest validated value
- Planner source
- Model name
- Validation status
- Quality-ready status

### Analysis Results

- Bar chart: `AnalysisResults[label]` by `AnalysisResults[value]`
- KPI card: `AnalysisResults[value]`
- Result table with rank, label, value, metric, and dimension

### AI Evidence

- Question
- Planner source
- Intent
- Metric
- Dimension
- Confidence
- Reasoning summary
- Approved tool steps

### Data Quality

- Ready status
- Duplicate rows
- Missing values
- Usable numeric ratio
- Warnings

### Source Documents

- Filename
- File type
- OCR status
- OCR languages
- Extraction strategy
- Extraction confidence

## 5. Refresh

After publishing a new result from Streamlit, select **Home > Refresh** in
Power BI Desktop. For a Power BI App, place the workbook in OneDrive or
SharePoint, publish the report to a workspace, configure refresh, and publish or
update the Power BI App.

## 6. Docker

Docker is intentionally excluded from the current project scope. InsightFlow AI,
Ollama, OCR tools, and Power BI Desktop run natively on Windows.
