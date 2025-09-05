from pathlib import Path
import numpy as np
import torch
import traceback
from Bio import SeqIO
from chai_lab.chai1 import run_inference

# ==== CONFIG ====
BATCH_SIZE = 1  # For two ligands, batch size of 1 is safer
RECYLES = 3
TIMESTEPS = 200
SEED = 42
FASTA_FILE = Path("two_hemes.fasta")
OUTPUT_DIR = Path("./outputs/chai_twoheme_predictions")
DELETE_TEMP_FASTA = True
# ================

def parse_fasta_for_triplets(fasta_file):
    records = list(SeqIO.parse(fasta_file, "fasta"))
    triplets = []
    for i in range(0, len(records), 3):
        protein = records[i]
        ligand1 = records[i + 1]
        ligand2 = records[i + 2]
        pname = protein.id.replace("protein|name=", "")
        lname1 = ligand1.id.replace("ligand|name=", "")
        lname2 = ligand2.id.replace("ligand|name=", "")
        triplets.append((pname, str(protein.seq), lname1, str(ligand1.seq), lname2, str(ligand2.seq)))
    return triplets

def run_triplet(protein_name, protein_seq, ligand1_name, ligand1_seq, ligand2_name, ligand2_seq, device):
    output_dir = OUTPUT_DIR / f"{protein_name}_{ligand1_name}_{ligand2_name}"
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"? Skipping {protein_name}_{ligand1_name}_{ligand2_name} (already exists)")
        return

    print(f"?? Running: {protein_name} + {ligand1_name} + {ligand2_name}")
    fasta_str = f">protein|name={protein_name}\n{protein_seq}\n"
    fasta_str += f">ligand|name={ligand1_name}\n{ligand1_seq}\n"
    fasta_str += f">ligand|name={ligand2_name}\n{ligand2_seq}\n"

    temp_fasta_path = OUTPUT_DIR / f"{protein_name}_{ligand1_name}_{ligand2_name}.fasta"
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

        print(f"? Completed: {protein_name}_{ligand1_name}_{ligand2_name}")
        for idx, rd in enumerate(candidates.ranking_data):
            print(f"Model {idx}: Score = {rd.aggregate_score}")

        score_file = output_dir / "scores.model_idx_2.npz"
        if score_file.exists():
            scores = np.load(score_file)
            print(f"Score details: {scores.files}")

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"? OOM on {protein_name}_{ligand1_name}_{ligand2_name}, skipping.")
            torch.cuda.empty_cache()
        else:
            print(f"? Unknown error on {protein_name}_{ligand1_name}_{ligand2_name}:\n{traceback.format_exc()}")
    except Exception:
        print(f"? Unexpected error:\n{traceback.format_exc()}")

    if DELETE_TEMP_FASTA and temp_fasta_path.exists():
        temp_fasta_path.unlink()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    triplets = parse_fasta_for_triplets(FASTA_FILE)

    print(f"?? Total triplets: {len(triplets)}. Running in batches of {BATCH_SIZE}.\n")

    for i in range(0, len(triplets), BATCH_SIZE):
        batch = triplets[i:i + BATCH_SIZE]
        print(f"\n=== Batch {i // BATCH_SIZE + 1} ===")
        for pname, pseq, lname1, lseq1, lname2, lseq2 in batch:
            # Ensure ligand names are unique even if the same ligand is used twice
            lname1_tagged = lname1
            lname2_tagged = lname2 if lname2 != lname1 else lname2 + "_2"
            run_triplet(pname, pseq, lname1_tagged, lseq1, lname2_tagged, lseq2, device)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
