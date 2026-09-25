def rtmar_shared_seed_mask(edge_index, n, d, mu, seed=1):
    """
    RT shared-seed missingness.

    Feature-level missingness with:
      - one shared seed per connected component;
      - all features start from the same local failure source;
      - feature-specific compact growth;
      - independent tie-breaking across features;
      - exact global missing rate floor(mu * n * d).

    Returns:
        torch.BoolTensor of shape [n, d], where True means missing.

    Uses only topology and randomness, never X or Y.
    """
    import numpy as np
    import torch
    from decimal import Decimal, ROUND_FLOOR
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rate = Decimal(str(mu))

    if n < 1 or d < 1 or not rate.is_finite() or not (0 <= rate <= 1):
        raise ValueError("Require n,d >= 1 and finite mu in [0,1]")
    if int(seed) != seed or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    edges = torch.as_tensor(edge_index).detach().cpu().numpy()

    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("edge_index must contain integers")
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("Invalid node index")

    # ---------------------------------------------------------
    # Random streams
    # ---------------------------------------------------------
    streams = np.random.SeedSequence(int(seed)).spawn(4)
    quota_rng, source_rng, tie_rng, component_rng = [
        np.random.default_rng(s) for s in streams
    ]

    # ---------------------------------------------------------
    # Undirected, unweighted graph without self-loops
    # ---------------------------------------------------------
    adj = coo_matrix(
        (np.ones(edges.shape[1], dtype=np.float64),
         (edges[0], edges[1])),
        shape=(n, n),
    ).tocsr()

    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1

    neighbors = [
        adj.indices[adj.indptr[i]:adj.indptr[i + 1]]
        for i in range(n)
    ]

    # ---------------------------------------------------------
    # Connected components
    # ---------------------------------------------------------
    n_components, labels = connected_components(
        adj, directed=False, return_labels=True
    )

    components = [
        np.flatnonzero(labels == c)
        for c in range(n_components)
    ]
    component_sizes = np.array([len(c) for c in components], dtype=int)

    # ---------------------------------------------------------
    # Helper: component allocation
    # ---------------------------------------------------------
    def allocate_over_components(total, rng):
        if total == 0:
            return np.zeros(n_components, dtype=int)

        exact = total * component_sizes / n
        quotas = np.floor(exact).astype(int)
        quotas = np.minimum(quotas, component_sizes)

        remaining = int(total - quotas.sum())

        frac = exact - np.floor(exact)
        random_tie = rng.random(n_components)

        order = sorted(
            range(n_components),
            key=lambda c: (frac[c], random_tie[c]),
            reverse=True,
        )

        while remaining > 0:
            changed = False
            for c in order:
                if quotas[c] < component_sizes[c]:
                    quotas[c] += 1
                    remaining -= 1
                    changed = True
                    if remaining == 0:
                        break
            if not changed:
                raise RuntimeError("Unable to allocate component quotas.")

        return quotas

    # ---------------------------------------------------------
    # Global budget -> feature budgets
    # ---------------------------------------------------------
    m = int(
        (rate * Decimal(n * d)).to_integral_value(rounding=ROUND_FLOOR)
    )
    q, remainder = divmod(m, d)
    feature_priority = quota_rng.permutation(d)

    feature_quotas = np.full(d, q, dtype=int)
    feature_quotas[feature_priority[:remainder]] += 1

    component_quotas = np.zeros((d, n_components), dtype=int)

    for j in range(d):
        component_quotas[j] = allocate_over_components(
            feature_quotas[j], component_rng
        )

    # ---------------------------------------------------------
    # ONE shared seed per connected component
    # ---------------------------------------------------------
    component_sources = np.array(
        [int(source_rng.choice(nodes)) for nodes in components],
        dtype=int,
    )

    # ---------------------------------------------------------
    # Feature-specific tie-breaking
    # ---------------------------------------------------------
    tie_priority = np.empty((d, n), dtype=int)

    for j in range(d):
        perm = tie_rng.permutation(n)
        tie_priority[j, perm] = np.arange(n)

    mask = np.zeros((n, d), dtype=bool)
    regions = {}

    # All active features in component c start from the SAME source.
    for j in range(d):
        for c in range(n_components):
            quota = component_quotas[j, c]

            if quota <= 0:
                continue

            source = int(component_sources[c])

            regions[(j, c)] = {source}
            mask[source, j] = True

    # ---------------------------------------------------------
    # Feature-specific compact growth
    # ---------------------------------------------------------
    while True:
        proposals = []

        for (j, c), region in regions.items():
            target = component_quotas[j, c]

            if len(region) >= target:
                continue

            candidates = set()

            for u in region:
                for v in neighbors[u]:
                    v = int(v)

                    if labels[v] == c and not mask[v, j]:
                        candidates.add(v)

            if not candidates:
                raise RuntimeError(
                    f"No candidate for feature {j}, component {c}."
                )

            best_v = None
            best_key = None

            for v in candidates:
                # Compact growth: prefer the candidate having the
                # largest number of neighbours already in C_j.
                same_feature_neighbors = sum(
                    bool(mask[int(u), j]) for u in neighbors[v]
                )

                key = (
                    same_feature_neighbors,
                    -tie_priority[j, v],
                )

                if best_key is None or key > best_key:
                    best_key = key
                    best_v = v

            proposals.append((j, c, best_v))

        if not proposals:
            break

        for j, c, v in proposals:
            if len(regions[(j, c)]) < component_quotas[j, c]:
                mask[v, j] = True
                regions[(j, c)].add(v)

    # ---------------------------------------------------------
    # Checks
    # ---------------------------------------------------------
    if int(mask.sum()) != m:
        raise RuntimeError(
            f"Expected {m} missing entries, obtained {mask.sum()}."
        )

    for j in range(d):
        if int(mask[:, j].sum()) != int(feature_quotas[j]):
            raise RuntimeError(f"Wrong quota for feature {j}.")

    return torch.from_numpy(mask)
