# Test requests and hardware wanted

Strata is built and measured on a few machines. Results from other hardware are welcome and are used: they tell us
what a setup really does and show where the engine picks something wrong. Write a report as described in
[COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md): the build, the card and PCIe width, the CPU and RAM speed, the
model file, the exact command, three runs per size, and what you did not test. Post it in an issue, or on the pull
request it is about.

This list is edited by the maintainers. An item stays until a report answers it.

| Wanted | Why | Example of an earlier answer |
|---|---|---|
| Two NVIDIA cards of different sizes (a large and a small one), in both card orders, with `--layer-split auto` | There is no unequal pair here to check the split planner; the card order and the small card's free memory changed the split | #880, #1238 |
| Three or four NVIDIA GPUs | No multi-GPU NVIDIA rig of that size here | #880 |
| Pascal (sm_61: Tesla P40, P100, GTX 10 series) | Older cards run on the fp32 path and can behave differently from newer ones | #875, #876 |
| A Linux host with 32 GB of RAM or less | The expert file tier and the resident mode depend on how much of the model fits in RAM | #1194 |
| A long soak (an hour or more of back-to-back requests) on a split or resident setup | Shows crashes, drift and heat that a short run does not | #848 |
| AMD RX 7000 series (RX 7900 XTX, 7900 XT, 7800 XT and similar, RDNA 3) on Linux or Windows | The RX 7000 builds have not been run on a real RX 7000 card by us; the Docker image for them is also untested on a card | none yet |
| Windows with an AMD card: the app or the Terminal, one model, one picture sent | The Windows AMD build has not been run on a real card, and neither has its picture reader (CPU under 24 GB, GPU from 24 GB) | none yet |
| A Mac with Apple silicon (macOS 15 or newer): the app or the Terminal with a real model | Mac builds have run on very few Macs. Tell us the Mac's memory, the model, the speed, and anything that failed. Pictures on the Mac (GPU reader) were run on one Mac mini M4 only: tell us if they work on yours. The signing script has not been run against Apple's real notary service yet, so a first-launch warning is expected and useful to hear about | none yet |
| One NVIDIA card of 12 to 24 GB (RTX 3060 to 4090) with 64 to 128 GB of RAM and a desktop CPU, DeepSeek with `--mmap-experts off`, on Linux and on Windows: the same short question with `STRATA_DS4_ONE_GPU=1` and without it | The one-GPU decode path (the PCIe lane, `docs/DEEPSEEK41.md`) was measured in this engine only on a cloud card with its memory capped and a 24-thread server CPU, where it was slower; Project Maya measured it faster on an RTX 3090 | none yet |

To add a request, open a pull request that adds a row, or tell the maintainers in an issue.
