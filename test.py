import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Assuming your class is in retrieval_system.py
from retrieval_system import MusicRetrievalSystem


def run_performance_test():
    # 1. Setup Parameters
    DIMENSION = 1024
    LIMIT = 5_000_000  # 5 Million entries

    # Generate dense checkpoints (1, 2, 5 series) for log-scale distribution
    CHECKPOINTS = []
    for i in range(1, 7):
        for multiplier in [1, 2, 5]:
            val = multiplier * (10 ** i)
            if val <= LIMIT:
                CHECKPOINTS.append(val)
    CHECKPOINTS = sorted(list(set(CHECKPOINTS)))

    DB_PATH = "benchmark_extreme_log.db"

    # Cleanup previous data
    if os.path.exists(DB_PATH): os.remove(DB_PATH)
    if os.path.exists("music.index"): os.remove("music.index")

    # Initialize System
    system = MusicRetrievalSystem(dimension=DIMENSION, db_path=DB_PATH)

    results_no_vec = []
    results_with_vec = []

    print(f"🚀 Starting Extreme Log-Log Benchmark")
    print(f"Target: {LIMIT} entries | Dimension: {DIMENSION}")

    last_checkpoint = 0

    with tqdm(total=LIMIT, desc="Data Ingestion") as pbar:
        for cp in CHECKPOINTS:
            num_to_add = cp - last_checkpoint

            # Batch generation to save memory and time
            inner_batch = 20000
            for i in range(0, num_to_add, inner_batch):
                current_chunk_size = min(inner_batch, num_to_add - i)
                vectors = np.random.random((current_chunk_size, DIMENSION)).astype('float32')

                for j in range(current_chunk_size):
                    global_idx = last_checkpoint + i + j
                    system.add_song(
                        vector=vectors[j],
                        track_id=f"T_{global_idx}",
                        clique_id=f"C_{global_idx // 10}"
                    )
                pbar.update(current_chunk_size)

            # --- Testing Phase ---
            test_query = np.random.random(DIMENSION).astype('float32')

            # Warm-up
            system.search(test_query, k=10, return_vectors=False)

            # 5 Trials: Metadata Only
            latencies_nv = []
            for _ in range(5):
                start = time.perf_counter()
                system.search(test_query, k=10, return_vectors=False)
                latencies_nv.append(time.perf_counter() - start)

            # 5 Trials: Metadata + Vector (Reconstruction)
            latencies_v = []
            for _ in range(5):
                start = time.perf_counter()
                system.search(test_query, k=10, return_vectors=True)
                latencies_v.append(time.perf_counter() - start)

            results_no_vec.append(np.mean(latencies_nv))
            results_with_vec.append(np.mean(latencies_v))
            last_checkpoint = cp

    # 2. Final Plotting (Log-Log Scale)
    plt.figure(figsize=(12, 8))

    # Plotting with Log-Log support
    plt.loglog(CHECKPOINTS, results_no_vec, 'o-', label='Search (Metadata Only)', linewidth=2, markersize=8)
    plt.loglog(CHECKPOINTS, results_with_vec, 's--', label='Search (Reconstruct Vector)', linewidth=2, markersize=8)

    # Customizing the Grid for Log scale
    plt.grid(True, which="both", ls="-", alpha=0.5)

    # Labels and Titles
    plt.xlabel('Database Size (N) - Log Scale', fontsize=12)
    plt.ylabel('Query Latency (Seconds) - Log Scale', fontsize=12)
    plt.title(f'Log-Log Performance Analysis (Scaling to 5M, Dim={DIMENSION})', fontsize=14)
    plt.legend()

    # Adding a reference line for O(N) complexity
    # We take the first measurement as a baseline for the O(N) trend
    if len(results_no_vec) > 1:
        base_x = CHECKPOINTS[0]
        base_y = results_no_vec[0]
        o_n_line = [base_y * (x / base_x) for x in CHECKPOINTS]
        plt.plot(CHECKPOINTS, o_n_line, 'k:', alpha=0.3, label='Theoretical O(N) Trend')
        plt.legend()

    # Save the output
    output_filename = 'log_log_scaling_5M.png'
    plt.savefig(output_filename, dpi=300)
    print(f"\n✅ Benchmark Complete!")
    print(f"Plot saved as: {output_filename}")
    plt.show()


if __name__ == "__main__":
    run_performance_test()