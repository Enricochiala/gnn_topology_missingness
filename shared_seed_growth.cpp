#include <cstdint>
#include <queue>
#include <vector>

struct Candidate {
    int count;
    uint16_t tie;
    int node;
    bool operator<(const Candidate &other) const {
        if (count != other.count) return count < other.count;
        return tie > other.tie;
    }
};

extern "C" int shared_seed_growth(int64_t n, int64_t d, int64_t components,
    const int64_t *indptr, const int64_t *indices, const int64_t *sources,
    const int64_t *sizes, const uint16_t *ties, uint16_t *ranks, int threads) {
    int failures = 0;
    #pragma omp parallel for num_threads(threads) reduction(+:failures) schedule(static)
    for (int64_t j = 0; j < d; ++j) {
        std::vector<int> counts(n, 0);
        std::vector<uint8_t> included(n, 0);
        const uint16_t *tie = ties + j*n;
        uint16_t *rank = ranks + j*n;
        for (int64_t c = 0; c < components; ++c) {
            std::priority_queue<Candidate> frontier;
            int u = static_cast<int>(sources[c]);
            int added = 0;
            while (true) {
                included[u] = 1;
                rank[u] = static_cast<uint16_t>(added++);
                for (int64_t k = indptr[u]; k < indptr[u+1]; ++k) {
                    int v = static_cast<int>(indices[k]);
                    if (!included[v]) {
                        ++counts[v];
                        frontier.push({counts[v], tie[v], v});
                    }
                }
                if (added == sizes[c]) break;
                while (!frontier.empty() && (included[frontier.top().node] ||
                    frontier.top().count != counts[frontier.top().node])) frontier.pop();
                if (frontier.empty()) { ++failures; break; }
                u = frontier.top().node;
                frontier.pop();
            }
        }
    }
    return failures;
}
