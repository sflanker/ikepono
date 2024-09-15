from datetime import datetime
import time

from ikepono.configuration import Configuration
from ikepono.reidentifier import Reidentifier


def main():
    start = time.time()
    dt = datetime.now()
    print(f"Training reidentifier at {dt}...")
    try:
        reidentifier = Reidentifier.for_training(Configuration("reidentifier_train_configuration.json"))
        training_time_in_seconds, best_mrr = reidentifier.train()
        print(f"Training time in seconds: {training_time_in_seconds}")
        print(f"Best MRR: {best_mrr}")
    finally:
        end = time.time()
        print(f"Training completed at {datetime.now()}. Total time: {end - start} seconds.")


if __name__ == '__main__':
    main()