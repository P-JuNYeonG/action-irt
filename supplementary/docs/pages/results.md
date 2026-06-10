---
layout: page
title: Results
permalink: /pages/results.html
---

# Empirical Results

## Data

The empirical analysis uses 14 PSTRE items from the U.S. PIAAC sample. After preprocessing, each respondent-item pair is represented as an ordered sequence of standardized action units and timestamps.

| Item | Task | Response type | N |
|------|------|---------------|---:|
| PS1-1 | Party Invitations, Part 1 | Polytomous (0-3) | 1,295 |
| PS1-2 | Party Invitations, Part 2 | Binary | 1,246 |
| PS1-3 | CD Tally | Binary | 1,272 |
| PS1-4 | Sprained Ankle, Part 1 | Binary | 1,255 |
| PS1-5 | Sprained Ankle, Part 2 | Binary | 1,302 |
| PS1-6 | Tickets | Binary | 1,282 |
| PS1-7 | Class Attendance | Polytomous (0-3) | 1,074 |
| PS2-1 | Club Membership, Part 1 | Binary | 1,274 |
| PS2-2 | Club Membership, Part 2 | Polytomous (0-3) | 1,170 |
| PS2-3 | Book Order | Binary | 1,238 |
| PS2-4 | Meeting Room | Polytomous (0-3) | 1,161 |
| PS2-5 | Reply All | Binary | 1,199 |
| PS2-6 | Locate Email | Polytomous (0-3) | 1,131 |
| PS2-7 | Lamp Return | Polytomous (0-3) | 1,230 |

## Selection Summary

Across all items, 126 of 2,025 action-item combinations were identified as important under the 95% HPD criterion for \(\Delta E_{jl}\).

<div class="summary-grid">
  <div class="summary-item"><strong>6.2%</strong>of evaluated action-item combinations selected</div>
  <div class="summary-item"><strong>67</strong>positive action effects</div>
  <div class="summary-item"><strong>59</strong>negative action effects</div>
  <div class="summary-item"><strong>D = 1</strong>latent dimension used in the empirical fit</div>
</div>

| Item | Total actions | Selected | Positive | Negative |
|------|--------------:|---------:|---------:|---------:|
| PS1-1 | 119 | 16 | 13 | 3 |
| PS1-2 | 141 | 10 | 5 | 5 |
| PS1-3 | 124 | 3 | 2 | 1 |
| PS1-4 | 47 | 7 | 6 | 1 |
| PS1-5 | 48 | 4 | 1 | 3 |
| PS1-6 | 182 | 16 | 8 | 8 |
| PS1-7 | 200 | 11 | 5 | 6 |
| PS2-1 | 156 | 10 | 5 | 5 |
| PS2-2 | 289 | 4 | 1 | 3 |
| PS2-3 | 70 | 2 | 2 | 0 |
| PS2-4 | 206 | 14 | 6 | 8 |
| PS2-5 | 112 | 10 | 3 | 7 |
| PS2-6 | 146 | 11 | 6 | 5 |
| PS2-7 | 185 | 8 | 4 | 4 |
| **Total** | **2,025** | **126** | **67** | **59** |

## Interpretation Framework

Selected actions are interpreted at two levels. First, selection indicates that an action's latent representation carries information about response accuracy beyond respondent ability and item difficulty. Second, when the action sequence is sufficiently rich, the sign and magnitude of \(\Delta E_{jl}\) can support process-level interpretation of the sequential context in which an action occurred.

| Type | Defining characteristic | Items |
|------|-------------------------|-------|
| Outcome-dominant | Most selected actions are final-choice operations | PS1-3, PS1-4, PS1-5, PS1-7, PS2-3 |
| Process-dominant | Selected actions span intermediate process steps | PS1-1, PS1-2, PS2-1, PS2-5, PS2-6 |
| Hybrid | Outcome-proximal and process-level actions coexist | PS1-6, PS2-2, PS2-4, PS2-7 |

## Representative Case Studies

### PS1-3: CD Tally

The CD Tally item is outcome-dominant. The model selects the correct combobox answer, a sorting operation, and an unnecessary interface switch. The result illustrates that the method can recover the final answer operation while also identifying a strategy-related action.

### PS1-1: Party Invitations, Part 1

The Party Invitations item is process-dominant. Sixteen actions are selected, including message viewing, dragging, dropping, and non-task-relevant exploration. The selected set supports interpretation of how correct and incorrect respondents differ across the solution process.

### PS2-7: Lamp Return

The Lamp Return item is hybrid. Selected actions cover both intermediate steps and final confirmatory actions in a multi-step return procedure. The later completion steps show positive partial effects, while some early procedural steps show negative ability-adjusted effects despite being performed mostly by correct respondents.
