# Data and attribution

This repo uses the **Bank Marketing** dataset from the UCI Machine Learning Repository (dataset id 222), file `bank-additional-full.csv`: 41,188 rows, 20 input columns, and the target `y` (subscribed to a term deposit).

- Source: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Citation: S. Moro, P. Rita and P. Cortez, *Bank Marketing*, UCI Machine Learning Repository, 2012. https://doi.org/10.24432/C5K306
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0). No changes are made to the data; the code only reads it.
- Not committed. `src/data.py` downloads the archive at runtime into `data/raw/` (gitignored) and checks the extracted CSV against a pinned SHA-256 (`74adfc57...afb4d8`) and the 41,188-row count. A mismatch stops the run.
- Related paper: S. Moro, P. Cortez and P. Rita, "A data-driven approach to predict the success of bank telemarketing", *Decision Support Systems*, 62, 2014.
