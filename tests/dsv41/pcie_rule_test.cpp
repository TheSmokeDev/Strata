// Strata. The DeepSeek V4.1 engine's PCIe lane rules (src/dsv41/ds4_pcie.hpp): the split of a token's CPU-lane experts
// between the CPU and the GPU, and the one-GPU decode path's gate and defaults (Project Maya #476, Windows: #567).
// No GPU, no model.
#include "ds4_pcie.hpp"

#include <cstdio>

namespace pcie = strata::ds4::pcie;

static int fails = 0;
#define CHECK(c)                                                  \
    do {                                                          \
        if (!(c)) {                                               \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++fails;                                              \
        }                                                         \
    } while (0)

static int bits(unsigned m) {
    int n = 0;
    for (; m != 0; m &= m - 1) ++n;
    return n;
}

int main() {
    // the split: no costs measured, nothing streams
    pcie::Costs none;
    CHECK(!none.ok());
    CHECK(pcie::split(none, 6, 0u, 16) == 0u);
    // a stream lane much faster than the CPU lane: it takes picks, never more than its slots
    pcie::Costs fast;
    fast.c = 1.0, fast.oc = 0.05, fast.p = 0.2, fast.op = 0.1, fast.g = 0.01, fast.disk = 5.0;
    CHECK(fast.ok());
    CHECK(bits(pcie::split(fast, 6, 0u, 16)) >= 1);
    CHECK(bits(pcie::split(fast, 6, 0u, 2)) <= 2);
    CHECK(pcie::split(fast, 6, 0u, 0) == 0u);
    // a stream lane far slower than the CPU lane (a fast CPU lane beside a narrow link): nothing streams, warm or cold
    pcie::Costs slow;
    slow.c = 0.13, slow.oc = 0.02, slow.p = 3.0, slow.op = 0.5, slow.g = 0.05, slow.disk = 0.0;
    for (int n = 1; n <= 8; ++n) CHECK(pcie::split(slow, n, 0u, 16) == 0u);
    CHECK(!pcie::streams_any(slow, 16, 6));
    CHECK(pcie::streams_any(fast, 16, 6));
    // the chosen split is never slower than the CPU lane alone
    for (int n = 1; n <= 8; ++n) {
        double t = 0;
        const unsigned m = pcie::split(fast, n, 0u, 16, &t);
        CHECK(t <= pcie::split_time(fast, n, 0u, 0u) + 1e-12);
        CHECK(t == pcie::split_time(fast, n, 0u, m));
    }
    CHECK(pcie::table(none, 16) == "0 0 0 0 0 0 0 0 0");

    // the one-GPU path's gate: one discrete NVIDIA GPU, a group of one
    pcie::OneGpuSetup one;
    one.devs = 1, one.groups = 1, one.members = 1;
    CHECK(pcie::one_gpu_on(one));
    pcie::OneGpuSetup s = one;
    s.hip = true;   // an AMD build
    CHECK(!pcie::one_gpu_on(s));
    s = one, s.devs = 2, s.groups = 2;   // two cards
    CHECK(!pcie::one_gpu_on(s));
    s = one, s.devs = 2, s.members = 2;   // a tensor-parallel pair
    CHECK(!pcie::one_gpu_on(s));
    s = one, s.integrated = true;   // a GPU sharing the PC's RAM
    CHECK(!pcie::one_gpu_on(s));
    s = one, s.lane_off = true;   // STRATA_DS4_STREAM=0
    CHECK(!pcie::one_gpu_on(s));
    s = one, s.asked = 0;   // STRATA_DS4_ONE_GPU=0: the engine as before
    CHECK(!pcie::one_gpu_on(s));
    s = one, s.asked = 1;   // (1 does not force it where it cannot run)
    CHECK(pcie::one_gpu_on(s));
    s.hip = true;
    CHECK(!pcie::one_gpu_on(s));
    s = one, s.v026 = true;   // DS4_DEFAULTS=026: the 0.2.6 defaults had no such path ...
    CHECK(!pcie::one_gpu_on(s));
    s.asked = 1;   // ... unless the setting asks for it
    CHECK(pcie::one_gpu_on(s));
    s = one, s.times_off = true;   // the times file: this GPU streamed nothing at an earlier start
    CHECK(!pcie::one_gpu_on(s));
    s.forced = true;   // DS4_STREAM_FORCE (tests) does not read the file's verdict
    CHECK(pcie::one_gpu_on(s));

    // the settings that follow the path unless given
    CHECK(pcie::follow(-1, true) && !pcie::follow(-1, false));
    CHECK(!pcie::follow(0, true) && pcie::follow(1, false));
    // the lane itself: on with the path, else only when asked; 0 always off
    CHECK(pcie::lane_on(-1, true) && !pcie::lane_on(-1, false));
    CHECK(pcie::lane_on(1, false) && pcie::lane_on(2, false));
    CHECK(!pcie::lane_on(0, true));
    // pin by need: on by default for the path on Windows with NVIDIA only (#567); the setting decides when given
    CHECK(pcie::pin_by_need_on(-1, true, false, true));
    CHECK(!pcie::pin_by_need_on(-1, false, false, true));   // Linux pins every copy
    CHECK(!pcie::pin_by_need_on(-1, true, true, true));     // AMD on Windows
    CHECK(!pcie::pin_by_need_on(-1, true, false, false));   // off the path
    CHECK(!pcie::pin_by_need_on(0, true, false, true));
    CHECK(pcie::pin_by_need_on(1, false, false, false));
    // the recent slots: 32 a GGUF layer with the landings, on the chunked prompt path; else none by default
    CHECK(pcie::recent_default(true, true, true, true) == 32);
    CHECK(pcie::recent_default(true, true, true, false) == 0);
    CHECK(pcie::recent_default(false, true, true, true) == 0);
    CHECK(pcie::recent_default(true, false, false, true) == 0);

    if (fails == 0) std::printf("ds4 PCIe lane rules: all passed\n");
    return fails == 0 ? 0 : 1;
}
