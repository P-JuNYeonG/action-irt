# Submission Checklist for Psychometrika

## Pre-Submission Checklist

### Manuscript Requirements
- [ ] Main manuscript ≤ 40 double-spaced pages (12pt font, including references/tables/figures)
- [ ] Abstract ≤ 200 words
- [ ] Keywords: 3–5
- [ ] APA 7th edition formatting
- [ ] All equations numbered and referenced in text
- [ ] All tables and figures referenced in text

### Anonymization (for peer review)
- [ ] No author names in headers, footers, or running heads
- [ ] No author-identifying information in body text or author notes
- [ ] Self-citations cited normally (not disguised)
- [ ] File metadata cleaned (check PDF properties for author name)

### Required Cover-Page Statements

#### Data Availability Statement
```
The analysis uses log process data from the OECD Programme for the 
International Assessment of Adult Competencies (PIAAC), Problem Solving 
in Technology-Rich Environments domain. PIAAC public-use data files are 
available from the OECD PIAAC Data Portal 
(https://www.oecd.org/skills/piaac/data/). All analysis code is publicly 
available at https://github.com/<username>/action-irt.
```

#### Financial Support
```
[TO BE COMPLETED — list all funding sources, grant numbers, and 
funding agencies. If none, state: "This research received no specific 
grant from any funding agency, commercial, or not-for-profit sectors."]
```

#### Competing Interests
```
The authors declare no competing interests.
```

#### Corresponding Author ORCID
```
[TO BE COMPLETED — ORCID iD of the corresponding author is required]
```

### File Preparation
- [ ] Cover page (separate file): title, authors, affiliations, required statements
- [ ] Anonymized manuscript (separate file): main text without identifying information
- [ ] Figures: high-resolution (≥ 300 dpi), separate files if required
- [ ] Tables: editable format (not images)
- [ ] Supplementary material / appendices: separate file(s) if needed
- [ ] PDF compiled from LaTeX source

### Code & Reproducibility
- [ ] GitHub repository public or ready to make public upon acceptance
- [ ] README with reproduction instructions
- [ ] Package versions documented (renv / requirements.txt)
- [ ] Random seeds recorded for all stochastic procedures
- [ ] Config file separates hyperparameters from code

### Final Checks
- [ ] All TODO comments in LaTeX resolved
- [ ] All citation keys have corresponding .bib entries
- [ ] No placeholder references remain
- [ ] Notation consistent throughout (check Table of Notation)
- [ ] Simulation settings in text match config.yaml values
- [ ] Number of important actions (126) consistent across all mentions
