from pathlib import Path
import numpy as np
import torch
import traceback
from Bio import SeqIO
from chai_lab.chai1 import run_inference

# ==== CONFIG ====
BATCH_SIZE = 1  
RECYLES = 3
TIMESTEPS = 200
SEED = 42
FASTA_FILE = Path("one_heme.fasta")
OUTPUT_DIR = Path("./outputs/chai_oneheme_predictions")
DELETE_TEMP_FASTA = True
# ===============

def parse_fasta_for_pairs(fasta_file):
    records = list(SeqIO.parse(fasta_file, "fasta"))
    pairs = []
    for i in range(0, len(records), 2):
        protein = records[i]
        ligand = records[i + 1]
        pname = protein.id.replace("protein|name=", "")
        lname = ligand.id.replace("ligand|name=", "")
        pairs.append((pname, str(protein.seq), lname, str(ligand.seq)))
    return pairs

def run_pair(protein_name, protein_seq, ligand_name, ligand_seq, device):
    output_dir = OUTPUT_DIR / f"{protein_name}_{ligand_name}"
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"? Skipping {protein_name}_{ligand_name} (already exists)")
        return

    print(f"?? Running: {protein_name} + {ligand_name}")
    fasta_str = f">protein|name={protein_name}\n{protein_seq}\n>ligand|name={ligand_name}\n{ligand_seq}\n"
    temp_fasta_path = OUTPUT_DIR / f"{protein_name}_{ligand_name}.fasta"
    temp_fasta_path.write_text(fasta_str)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        candidates = run_inference(
            fasta_file=temp_fasta_path,
            output_dir=output_dir,
            num_trunk_recycles=RECYLES,
            num_diffn_timesteps=TIMESTEPS,
            seed=SEED,
            device=device,
            use_esm_embeddings=True,
        )

        print(f"? Completed: {protein_name}_{ligand_name}")
        for idx, rd in enumerate(candidates.ranking_data):
            print(f"Model {idx}: Score = {rd.aggregate_score}")

        score_file = output_dir / "scores.model_idx_2.npz"
        if score_file.exists():
            scores = np.load(score_file)
            print(f"Score details: {scores.files}")

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"? OOM on {protein_name}_{ligand_name}, skipping.")
            torch.cuda.empty_cache()
        else:
            print(f"? Unknown error on {protein_name}_{ligand_name}:\n{traceback.format_exc()}")
    except Exception:
        print(f"? Unexpected error:\n{traceback.format_exc()}")

    if DELETE_TEMP_FASTA and temp_fasta_path.exists():
        temp_fasta_path.unlink()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pairs = parse_fasta_for_pairs(FASTA_FILE)

    print(f"?? Total pairs: {len(pairs)}. Running in batches of {BATCH_SIZE}.\n")

    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i:i + BATCH_SIZE]
        print(f"\n=== Batch {i//BATCH_SIZE + 1} ===")
        for pname, pseq, lname, lseq in batch:
            run_pair(pname, pseq, lname, lseq, device)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
