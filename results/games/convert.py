import argparse
import json
import os
import pickle

import numpy as np
from sklearn.decomposition import PCA


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def convert(file_name, input_dir="json", output_dir="handled", n_components=512):
    input_path = os.path.join(input_dir, f"{file_name}.jsonl")
    records = read_jsonl(input_path)
    by_id = {str(record["item_id"]): record["hidden_state"] for record in records}
    embeddings = np.asarray([by_id[str(i)] for i in range(1, len(records) + 1)], dtype=np.float32)

    pca = PCA(n_components=n_components, svd_solver="auto", random_state=42)
    reduced = pca.fit_transform(embeddings).astype(np.float32)
    if not np.all(np.isfinite(reduced)):
        num_nan = int(np.isnan(reduced).sum())
        num_inf = int(np.isinf(reduced).sum())
        raise ValueError(f"Detected invalid PCA values: NaN={num_nan}, Inf={num_inf}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{file_name}_{n_components}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(reduced.tolist(), f)
    print(f"Saved {output_path} with shape {reduced.shape}")


def main():
    parser = argparse.ArgumentParser(description="Convert item embedding JSONL to a PCA-reduced pickle file.")
    parser.add_argument("file_name", help="Input file stem under the json directory, without .jsonl")
    parser.add_argument("--input-dir", default="json")
    parser.add_argument("--output-dir", default="handled")
    parser.add_argument("--n-components", type=int, default=512)
    args = parser.parse_args()
    convert(args.file_name, args.input_dir, args.output_dir, args.n_components)


if __name__ == "__main__":
    main()
