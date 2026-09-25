# 08 — Data Preparation Checklist (do before M03/M04)

| # | Task | Output | Owner |
|---|---|---|---|
| 1 | Export Stage 2 predictions on train/val/test | `data/stage2_pairs/*.jsonl` | You |
| 2 | Export per-word confidences for test pages | `data/pages/*.json` | You |
| 3 | Build seed ECE term list (VTU syllabus, textbook indexes) | `ece_terms.txt` | You (+ agent to normalise) |
| 4 | Get general English wordlist with frequencies | `general_dictionary.txt` | You |
| 5 | Build phrase corpus from answer dataset + textbooks | `domain_phrases.jsonl` | You (+ agent script) |
| 6 | Hand-label ~50 corrupted terms/phrases with correct targets | `tests/fixtures/labeled_errors.yaml` | You |
| 7 | Confirm licences for any textbook text used | note in README | You |

Until (1)–(5) exist, milestones use clearly labelled **synthetic fixtures** under `tests/fixtures/`; the agent must not present fixture-based numbers as results.
