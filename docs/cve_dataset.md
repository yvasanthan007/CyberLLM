# CVE Dataset

## Raw Dataset

Source:
NVD API

Location:
dataset/raw/cve/

Format:
JSON

---

## Processed Dataset

Location:
dataset/processed/cve_clean.jsonl

Fields:

- cve_id
- description
- published
- last_modified
- severity
- cvss_score
- references

---

## Instruction Dataset

Location:
dataset/instruction/cve_instruction.jsonl

Fields:

- instruction
- input
- output

Each CVE produces multiple instruction examples.