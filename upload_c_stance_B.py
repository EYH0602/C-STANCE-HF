from pathlib import Path
from typing import Dict

import fire
from datasets import DatasetDict, load_dataset


def _build_data_files(base_dir: Path) -> Dict[str, str]:
    csv_map = {
        "train": "train.csv",
        "validation": "validation.csv",
        "test": "test.csv",
    }
    data_files = {}
    for split, filename in csv_map.items():
        path = base_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Expected CSV file for split '{split}' at {path}")
        data_files[split] = str(path)
    return data_files


def main(
    c_stance_B_dir: str = "data/subtaskB_combined",
    hub_repo_id: str = "yfhe/C-STANCE-B",
    private: bool = False,
) -> None:
    """Load deduplicated C-STANCE subtask B splits and upload them to HuggingFace Hub."""
    base_dir = Path(c_stance_B_dir).expanduser().resolve()
    if not base_dir.is_dir():
        raise ValueError(f"Expected directory at {base_dir}")

    dataset = load_dataset("csv", data_files=_build_data_files(base_dir))
    if not isinstance(dataset, DatasetDict):
        raise RuntimeError("Expected a DatasetDict with named splits.")
    dataset.push_to_hub(hub_repo_id, private=private)


if __name__ == "__main__":
    fire.Fire(main)
