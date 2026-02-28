from train import train
from test import test
from evaluation import evaluation
from code_config.parser import parse_json_file

import os


def main():
    config_root = "./file_config"

    configs = {
        "train.json": parse_json_file(os.path.join(config_root, "train.json")),
        "test.json": parse_json_file(os.path.join(config_root, "test.json")),
        "evaluation.json": parse_json_file(os.path.join(config_root, "evaluation.json")),
    }

    do_train = True
    do_test = True
    do_eval = True

    train_output = None
    test_output = None

    if do_train:
        print("Starting training...")
        train_output = train(status_config=configs["train.json"])
        print("Training completed.")

    if do_test:
        print("Starting testing...")
        test_output = test(status_config=configs["test.json"], common_config=train_output)
        print("Testing completed.")

    if do_eval:
        print("Starting evaluation...")
        eval_output = evaluation(status_config=configs["evaluation.json"], common_config=test_output)
        print("Evaluation completed.")


if __name__ == "__main__":
    main()