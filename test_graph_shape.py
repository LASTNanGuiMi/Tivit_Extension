import numpy as np
from PIL import Image


def get_optimal_order(n):
    if n < 0:
        raise ValueError(f"Number of signal channels must be non-negative, got {n}.")
    if n <= 1:
        return list(range(n))

    total_pairs = n * (n - 1) // 2
    visited_pairs = set()
    id_list = [0]

    while len(visited_pairs) < total_pairs:
        current = id_list[-1]
        next_id = None

        for candidate in range(n):
            if candidate == current:
                continue
            pair = tuple(sorted((current, candidate)))
            if pair not in visited_pairs:
                next_id = candidate
                break

        if next_id is None:
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) not in visited_pairs:
                        next_id = i if current != i else j
                        break
                if next_id is not None:
                    break

        pair = tuple(sorted((current, next_id)))
        visited_pairs.add(pair)
        id_list.append(next_id)

    return id_list


def build_multicolumn_graph(signals, id_list):
    rows = []
    for k, signal_id in enumerate(id_list):
        left = signals[id_list[k - 1]]
        center = signals[signal_id]
        right = signals[id_list[(k + 1) % len(id_list)]]
        rows.append(np.concatenate([left, center, right], axis=0))

    return np.stack(rows, axis=0)


def generate_activity_graph(signals, mode="multicolumn"):
    if mode != "multicolumn":
        raise ValueError(f"Unsupported mode {mode}.")

    id_list = get_optimal_order(signals.shape[0])
    return build_multicolumn_graph(signals, id_list)


# Only use numpy/PIL to simulate preprocess_graph shape changes. Do not import torch.
# Verify every shape step from (B, n, T) to (B, 3, img_size, img_size).
B, n, T, img_size = 4, 6, 128, 224

# Simulate one batch.
signals = np.random.randn(B, n, T)

results = []
for i in range(B):
    # Generate activity graph.
    graph = generate_activity_graph(signals[i], mode="multicolumn")
    print(f"sample {i} graph shape: {graph.shape}")

    # Normalize.
    g_min, g_max = graph.min(), graph.max()
    graph = (graph - g_min) / (g_max - g_min + 1e-8)

    # Simulate resize with simple PIL interpolation to (img_size, img_size).
    img = Image.fromarray((graph * 255).astype(np.uint8))
    img = img.resize((img_size, img_size))
    arr = np.array(img)

    # Copy to 3 channels.
    arr = np.stack([arr, arr, arr], axis=0)
    results.append(arr)

results = np.stack(results, axis=0)
print(f"final batch shape: {results.shape}")
assert results.shape == (B, 3, img_size, img_size), "shape error"
print("shape verification passed")
