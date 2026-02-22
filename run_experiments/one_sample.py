import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run a single instance of basic_parser.py')


    parser.add_argument('--cuda', default=0, type=int, help='CUDA device ID')
    parser.add_argument('--task', default='modular_arithmetic', type=str, help='Task to run')
    parser.add_argument('--architecture', default='rnn', type=str, help='Model architecture')
    parser.add_argument('--seed', default=3, type=int, help='Random seed')
    parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate')
    parser.add_argument('--training_steps', default=100000, type=int, help='Number of training steps')

    args = parser.parse_args()

    save_dir = '/media/hdd3/bong/tmp'

    folder_name = f'baseline-{args.lr}-{args.training_steps}'

    print(f"\n#### Run Single Experiment")
    print(f"Task: {args.task} | Architecture: {args.architecture} | Seed: {args.seed}")

    cmd = (
        f"python basic_parser.py "
        f"--cuda {args.cuda} "
        f"--folder_name {folder_name} "
        f"--training_steps {args.training_steps} "
        f"--seed {args.seed} "
        f"--task {args.task} "
        f"--architecture {args.architecture} "
        f"--lr {args.lr} "
        f"--save_dir {save_dir} "
    )

    print(f"\n[Executing Command]\n{cmd}\n")

    os.system(cmd)


if __name__ == "__main__":
    main()