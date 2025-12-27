import argparse
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from attack_simulator import run_attack_simulation, ATTACK_MAPPING
from defense_simulator import run_defense_simulation, DEFENSE_MAPPING

def main():
    # Priority: Command line args > Environment variables > Defaults
    env_mode = os.getenv("mode")
    env_method = os.getenv("method")
    env_json_dir = os.getenv("json_dir", "./json_file")
    env_output_path = os.getenv("OUTPUT_PATH", "./output")
    env_input_path = os.getenv("INPUT_PATH", "./input")

    parser = argparse.ArgumentParser(description="NUDT Poisoning Attack & Defense System")
    parser.add_argument("--mode", type=str, choices=["attack", "defense"], default=env_mode)
    parser.add_argument("--method", type=str, default=env_method)
    parser.add_argument("--json_dir", type=str, default=env_json_dir)
    parser.add_argument("--output_path", type=str, default=env_output_path)
    parser.add_argument("--input_path", type=str, default=env_input_path)

    args, unknown = parser.parse_known_args()

    # If mode is not provided as an argument, check if it was first positional (compatibility)
    if not args.mode and len(sys.argv) > 1 and sys.argv[1] in ["attack", "defense"]:
        args.mode = sys.argv[1]

    if not args.mode:
        print("Error: 'mode' (attack or defense) must be specified via environment variable or --mode.")
        parser.print_help()
        sys.exit(1)

    if not args.method:
        print("Error: 'method' must be specified via environment variable or --method.")
        sys.exit(1)

    if args.mode == "attack":
        run_attack_simulation(args.method, args.json_dir, args.output_path, input_dir=args.input_path)
    elif args.mode == "defense":
        run_defense_simulation(args.method, args.json_dir, args.output_path, input_dir=args.input_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
