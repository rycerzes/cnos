"""Download BOP classic datasets from Hugging Face Hub."""
from omegaconf import DictConfig, OmegaConf
import logging
import os
import glob
import hydra
from huggingface_hub import snapshot_download

# set level logging
logging.basicConfig(level=logging.INFO)


def run_download(dataset_name: str, local_dir: str) -> None:
    """Download and extract a dataset from Hugging Face Hub."""
    os.makedirs(local_dir, exist_ok=True)
    logging.info(f"Saving dataset to {local_dir}")

    # Download from Hugging Face (exclude train_pbr to save space)
    snapshot_download(
        repo_id=f"bop-benchmark/{dataset_name}",
        repo_type="dataset",
        local_dir=local_dir,
        ignore_patterns=["*train_pbr*"],
    )
    logging.info(f"Dataset downloaded to {local_dir}")

    # Extract all zip files
    logging.info("Extracting zip files...")
    zip_files = glob.glob(os.path.join(local_dir, "*.zip"))
    
    for zip_path in zip_files:
        zip_file = os.path.basename(zip_path)
        
        # Skip bop19 if test_all exists (test_all is more complete)
        if "bop19" in zip_file:
            test_all_file = zip_path.replace("bop19", "all")
            if os.path.exists(test_all_file):
                logging.info(f"Skipping {zip_file} as test_all exists")
                continue
        
        if "base" in zip_file:
            # Extract base files (camera params, etc.) with -j to flatten
            unzip_cmd = f"unzip -q -o -j {zip_path} -d {local_dir}"
        else:
            unzip_cmd = f"unzip -q -o {zip_path} -d {local_dir}"
        
        logging.info(f"Extracting {zip_file}...")
        os.system(unzip_cmd)
    
    logging.info(f"Extraction complete for {dataset_name}")


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="download",
)
def download(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    all_datasets = [
        "lmo",
        "tless",
        "tudl",
        "icbin",
        "itodd",
        "hb",
        "ycbv",
    ]
    
    # If dataset_name is specified, download only that one
    if cfg.dataset_name:
        datasets = [cfg.dataset_name]
    else:
        datasets = all_datasets
    
    for dataset_name in datasets:
        logging.info(f"Downloading {dataset_name} from Hugging Face Hub")
        local_dir = os.path.join(cfg.data.root_dir, dataset_name)
        run_download(dataset_name, local_dir)
        logging.info("---" * 50)
    
    logging.info("All datasets downloaded successfully!")


if __name__ == "__main__":
    download()
