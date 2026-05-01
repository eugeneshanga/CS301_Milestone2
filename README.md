# CS301 Milestone 2 – NYC Evictions Analysis

This project analyzes eviction patterns across New York City boroughs using eviction records and demographic data. The goal is to understand how demographic and housing characteristics relate to eviction rates.

## Project Contents
- `Copy_of_CS301_Milestone2_Evictions_Roadmap.ipynb` – Full analysis notebook
- `run_analysis.py` – Script that runs analysis and generates output
- `Evictions_20260428.csv` – Eviction dataset
- `demo_2016acs5yr_nyc.xlsx` – Demographic dataset
- `Dockerfile` – Container setup
- `requirements.txt` – Python dependencies
- `output/` – Folder where generated visualizations are saved

---

## Running with Docker

### Build the container
```
docker build -t cs301-evictions .
docker run -v $(pwd)/output:/app/output cs301-evictions
```
