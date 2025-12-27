import argparse
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from attack_simulator import run_attack_simulation, ATTACK_MAPPING
from defense_simulator import run_defense_simulation, DEFENSE_MAPPING

def main():
    parser = argparse.ArgumentParser(description="NUDT Poisoning Attack & Defense System")
    subparsers = parser.add_subparsers(dest="mode", help="Mode: attack or defense")

    # Attack subparser
    attack_parser = subparsers.add_parser("attack", help="Run poisoning attack")
    attack_parser.add_argument("--method", type=str, required=True, choices=list(ATTACK_MAPPING.keys()))
    attack_parser.add_argument("--json_dir", type=str, default="./json_file")
    attack_parser.add_argument("--output_path", type=str, default="./output")
    attack_parser.add_argument("--input_path", type=str, default="./input")

    # Defense subparser
    defense_parser = subparsers.add_parser("defense", help="Run poisoning defense")
    defense_parser.add_argument("--method", type=str, required=True, choices=list(DEFENSE_MAPPING.keys()))
    defense_parser.add_argument("--json_dir", type=str, default="./json_file")
    defense_parser.add_argument("--output_path", type=str, default="./output")
    defense_parser.add_argument("--input_path", type=str, default="./input")

    args = parser.parse_args()

    if args.mode == "attack":
        run_attack_simulation(args.method, args.json_dir, args.output_path, input_dir=args.input_path)
    elif args.mode == "defense":
        run_defense_simulation(args.method, args.json_dir, args.output_path, input_dir=args.input_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
