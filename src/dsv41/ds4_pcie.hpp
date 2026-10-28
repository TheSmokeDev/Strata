// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.33).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE): the PCIe lane (Maya #312, #329) and the one-GPU
// decode path's gate (RubenTorney's mw00/project-maya#476, Windows: #567); changes are in the git history of this file.
// src/dsv41/ds4_pcie.hpp - the decode path's PCIe lane for strata-dsv41 (the host side's arithmetic; ds4_model.cu runs it).
//
// A decoded token's routed experts that VRAM does not hold were all computed by the CPU lane.  As GLM's maya does
// (glm_fast_path.cu's lane_plan), some of them can instead be read into pinned memory, copied over PCIe into a few
// staging slots of the GPU and computed there by the same expert kernels as the VRAM ones - while the CPU lane computes
// the rest - so the two lanes finish together.  Which ones is decided per request from this PC's measured costs: the
// CPU lane's time an expert of the layer's format (at its thread count), the stream lane's (the read into pinned
// memory and the copy), the GPU's time an expert after the wait, and a read from the disk for experts not in RAM.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace strata::ds4::pcie {

/// One format class's measured times on this PC (ms): the CPU lane's an expert and a call; the stream lane's an expert
/// (its read into pinned memory and its copy to the GPU) and a request (the wait for the copies); the GPU's an expert
/// streamed (its gate / up and down after the wait); a read of one expert from the disk (not in RAM).
struct Costs {
    double c = 0, oc = 0, p = 0, op = 0, g = 0, disk = 0;
    bool ok() const { return c > 0 && p > 0; }
};

/// The time a request takes with the picks in `mask` streamed and the rest on the CPU lane: the slower lane, then the
/// streamed experts on the GPU.  Bit i of `cold`: pick i is not in RAM (both lanes read it from the disk first; the
/// stream lane's copy of it overlaps the read but the last part).
inline double split_time(const Costs& k, int n, unsigned cold, unsigned mask) {
    double tc = 0, tp = 0;
    int s = 0;
    for (int i = 0; i < n; ++i) {
        const bool cd = ((cold >> i) & 1u) != 0;
        if ((mask >> i) & 1u) {
            tp += cd ? k.disk + k.p / 3.0 : k.p;
            ++s;
        } else {
            tc += k.c + (cd ? k.disk : 0.0);
        }
    }
    if (s < n) tc += k.oc;
    if (s > 0) tp += k.op;
    return std::fmax(tc, tp) + k.g * s;
}

/// The split of n picks (n <= 8): the set of them the stream lane takes - at most max_s (its staging slots) - that
/// minimises split_time; a tie keeps the fewer streamed (then the lower mask).  Returns the mask; *t its time.
inline unsigned split(const Costs& k, int n, unsigned cold, int max_s, double* t = nullptr) {
    unsigned best = 0;
    double bt = split_time(k, n, cold, 0u);
    int bs = 0;
    if (k.ok() && max_s > 0)
        for (unsigned m = 1; m < (1u << n); ++m) {
            int s = 0;
            for (unsigned v = m; v != 0; v &= v - 1) ++s;
            if (s > max_s) continue;
            const double tm = split_time(k, n, cold, m);
            if (tm < bt - 1e-9 || (tm <= bt + 1e-9 && s < bs)) {
                best = m;
                bt = tm;
                bs = s;
            }
        }
    if (t != nullptr) *t = bt;
    return best;
}

/// How many of n warm picks the split streams, for n = 0..8 ("0 0 1 1 2 2 3 3 4": the log's table)
inline std::string table(const Costs& k, int max_s, int n_max = 8) {
    std::string s;
    for (int n = 0; n <= n_max; ++n) {
        const unsigned m = split(k, n, 0u, max_s);
        int c = 0;
        for (unsigned v = m; v != 0; v &= v - 1) ++c;
        s += (n ? " " : "") + std::to_string(c);
    }
    return s;
}

/// Whether the split streams anything for some n <= n_max picks, all of them warm or all cold
inline bool streams_any(const Costs& k, int max_s, int n_max) {
    for (int n = 1; n <= n_max && n <= 8; ++n)
        if (split(k, n, 0u, max_s) != 0u || split(k, n, (1u << n) - 1u, max_s) != 0u) return true;
    return false;
}

/// The one-GPU decode path's gate (ds4_model.cu's one_gpu_path): the setup RubenTorney's mw00/project-maya#476 was
/// measured on - one discrete NVIDIA GPU, a tensor-parallel group of one - on Linux and (#567) on Windows.  An AMD build,
/// several GPUs or groups, an integrated GPU: off.  Off too where the lane is off (STRATA_DS4_STREAM=0), where the times
/// file says this GPU's lane streamed nothing at an earlier start (unless DS4_STREAM_FORCE), with STRATA_DS4_ONE_GPU=0,
/// and under DS4_DEFAULTS=026 (Strata's 0.2.6 defaults, which had no such path) unless STRATA_DS4_ONE_GPU=1 asks for it.
struct OneGpuSetup {
    bool hip = false;          // an AMD build (STRATA_USE_HIP)
    int devs = 0;              // the GPUs the engine runs on
    int groups = 0;            // its tensor-parallel groups
    int members = 0;           // the first group's members
    bool integrated = false;   // the GPU shares the PC's RAM
    bool lane_off = false;     // STRATA_DS4_STREAM=0 (or DS4_STREAM=0)
    bool times_off = false;    // the times file's "<gpu> | lane" says "off"
    bool forced = false;       // DS4_STREAM_FORCE (tests: the timing plans nothing, the file is not read)
    int asked = -1;            // STRATA_DS4_ONE_GPU: -1 not set, 0 off, 1 on where it can run (over DS4_DEFAULTS=026)
    bool v026 = false;         // DS4_DEFAULTS=026
};
inline bool one_gpu_on(const OneGpuSetup& s) {
    if (s.hip || s.asked == 0 || (s.v026 && s.asked != 1)) return false;
    if (s.devs != 1 || s.groups != 1 || s.members != 1 || s.integrated) return false;
    if (s.lane_off) return false;
    return s.forced || !s.times_off;
}
/// A setting that follows the path unless given (-1: not set): DS4_STREAM_WIN, DS4_LAND and the like
inline bool follow(int set, bool path) { return set < 0 ? path : set != 0; }
/// The PCIe lane itself in Strata: on by default where the one-GPU path's gate holds (where it was measured), elsewhere
/// only when STRATA_DS4_STREAM (DS4_STREAM) asks for it (1 on, 2 the slots kept and nothing streamed: tests); 0 off
inline bool lane_on(int set, bool one) { return set < 0 ? one : set != 0; }
/// Pin by need (the copies pinned a layer at a time, the most-used first): STRATA_DS4_PIN_BY_NEED when given, else on
/// for the one-GPU path on Windows with NVIDIA (#567: WDDM lets a GPU lock about half the RAM, so pinning every copy fails)
inline bool pin_by_need_on(int set, bool windows, bool hip, bool one) {
    if (set >= 0) return set != 0;
    return windows && !hip && one;
}
/// The recent slots a layer gets by default (DS4_RECENT unset): a GGUF's 32 while streamed experts land in them
/// (DS4_LAND) and the prompts run in chunks, else none (Maya also gives a pack whose experts are not read from its files
/// 32 by default, #344; in Strata DS4_RECENT=N asks for them there)
inline int recent_default(bool chunked, bool file_experts, bool gguf, bool land) {
    if (!chunked) return 0;
    if (gguf && file_experts) return land ? 32 : 0;
    return 0;
}

/// The times file (DS4_LANE_CAL, by default <pack>/lane-times.txt): a line a key, "key<TAB>values".  get: the values
/// of `key` (empty: none); put: key's line replaced by vals (empty vals: dropped), through a temporary file.
inline std::string cal_get(const std::string& file, const std::string& key) {
    std::string out;
    if (FILE* f = std::fopen(file.c_str(), "r")) {
        char line[4096];
        while (std::fgets(line, sizeof line, f)) {
            const char* tab = std::strrchr(line, '\t');
            if (tab == nullptr || std::string(line, (size_t) (tab - line)) != key) continue;
            out = tab + 1;
            while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
        }
        std::fclose(f);
    }
    return out;
}
template <class Replace>
inline void cal_put(const std::string& file, const std::string& key, const std::string& vals, Replace replace) {
    std::string keep;
    if (FILE* f = std::fopen(file.c_str(), "r")) {
        char line[4096];
        while (std::fgets(line, sizeof line, f)) {
            const char* tab = std::strrchr(line, '\t');
            if (tab == nullptr || std::string(line, (size_t) (tab - line)) != key) keep += line;
        }
        std::fclose(f);
    }
    if (!vals.empty()) keep += key + "\t" + vals + "\n";
    const std::string tmp = file + ".tmp";
    if (FILE* f = std::fopen(tmp.c_str(), "w")) {
        const bool ok = std::fputs(keep.c_str(), f) >= 0;
        if (std::fclose(f) == 0 && ok && replace(tmp, file)) return;
        std::remove(tmp.c_str());
    }
}

}  // namespace strata::ds4::pcie
