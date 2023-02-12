---
bibliography: [ref.bib]
nocite: "@*"
---

pandoc -t markdown_strict --citeproc bib-test.md -o bib-test-output.md --bibliography ref.bib

