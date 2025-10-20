from os.path import join, abspath, isdir
import fire
from datasets.load import load_dataset


def main(c_stance_A_dir: str = "data/subtaskA"):
    """load C-STANCE subtaskA from local CSV and upload to HuggingFace Hub"""
    assert isdir(c_stance_A_dir)
    c_stance_A_dir = abspath(c_stance_A_dir)

    data_files = {
        "train": join(c_stance_A_dir, "raw_train_all_onecol.csv"),
        "test": join(c_stance_A_dir, "raw_test_all_onecol.csv"),
        "validation": join(c_stance_A_dir, "raw_val_all_onecol.csv"),
    }

    dataset = load_dataset("csv", data_files=data_files)
    dataset.push_to_hub("yfhe/C-STANCE-A")


if __name__ == "__main__":
    fire.Fire(main)
