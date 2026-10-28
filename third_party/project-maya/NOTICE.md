# Project Maya (maya-ds4), DeepSeek-V4.1 engine

The files under `src/dsv41/`, `include/strata/dsv41/`, `tests/dsv41/` and `tools/dsv41/` come from
https://github.com/mw00/project-maya (release v1.1.17, commit 8d9a9b5d), its DeepSeek engine `src/ds4`, `tests/ds4_*`,
`tools/ds4` and the kernel sources it builds on (`exl3`, `sampler`). Project Maya is itself built on Strata.

Copyright (c) 2026 Peasant Smith (mw00) and the Project Maya contributors, including needmorevram.
MIT License, the full text is in `LICENSE` beside this file. The notice stays with every copy.

Changes made while porting into Strata are listed in the git history of the `src/dsv41` directory
(the first commit of each file is the unmodified upstream copy).

Later changes taken from Project Maya releases v1.1.18 to v1.1.26 (the same approach re-written onto Strata's engine,
for example the long-prompt layer skip, the thinking budget, conversation slots) are covered by the same copyright
and permission notice, in `LICENSE` beside this file.

The one-GPU decode path in `src/dsv41/ds4_pcie.cuh`, `src/dsv41/ds4_pcie.hpp` and `src/dsv41/ds4_model.cu` comes from
Project Maya 1.1.33 (commit 88a746c) with the Windows change of mw00/project-maya#567 (commit 43b8a6c): the PCIe lane
(v1.1.18, #312; the whole-step check, #329), the recent slots (#344), RubenTorney's in-place pulls, streamed verify
windows, landings and device clock (#476), and the gate that keeps the path where it streams (#496). The same
copyright and permission notice covers it.
