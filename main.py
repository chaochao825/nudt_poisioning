import argparse
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from attack_simulator import run_attack_simulation, ATTACK_CHARACTERISTICS
from defense_simulator import run_defense_simulation, DEFENSE_CHARACTERISTICS

def main():
    # Helper to get env var with type conversion
    def get_env(key, default, type_func=str):
        val = os.getenv(key)
        if val is None:
            return default
        try:
            if type_func == bool:
                return val.lower() in ("true", "1", "yes")
            return type_func(val)
        except:
            return default

    # Read all from environment first
    env_mode = get_env("mode", None)
    env_method = get_env("method", None)
    env_json_dir = get_env("json_dir", "./json_file")
    env_output_path = get_env("OUTPUT_PATH", "./output")
    env_input_path = get_env("INPUT_PATH", "./input")
    
    env_poison_rate = get_env("poison_rate", 0.1, float)
    env_trigger_size = get_env("trigger_size", 3, int)
    env_target_label = get_env("target_label", 0, int)
    env_epochs = get_env("epochs", 2, int)
    env_batch_size = get_env("batch_size", 32, int)
    env_train_subset = get_env("train_subset", 500, int)
    env_test_subset = get_env("test_subset", 100, int)
    
    env_sensitivity = get_env("sensitivity", 0.5, float)
    env_threshold = get_env("threshold", 0.5, float)
    env_iterations = get_env("iterations", 100, int)

    parser = argparse.ArgumentParser(description="NUDT Poisoning Attack & Defense System")
    
    # Arguments
    parser.add_argument("--mode", type=str, choices=["attack", "defense"], default=env_mode)
    parser.add_argument("--method", type=str, default=env_method)
    parser.add_argument("--json_dir", type=str, default=env_json_dir)
    parser.add_argument("--output_path", type=str, default=env_output_path)
    parser.add_argument("--input_path", type=str, default=env_input_path)
    parser.add_argument("--train_subset", type=int, default=env_train_subset)
    parser.add_argument("--test_subset", type=int, default=env_test_subset)
    parser.add_argument("--poison_rate", type=float, default=env_poison_rate)
    parser.add_argument("--trigger_size", type=int, default=env_trigger_size)
    parser.add_argument("--target_label", type=int, default=env_target_label)
    parser.add_argument("--epochs", type=int, default=env_epochs)
    parser.add_argument("--batch_size", type=int, default=env_batch_size)
    parser.add_argument("--learning_rate", type=float, default=get_env("learning_rate", 0.001, float))
    parser.add_argument("--sensitivity", type=float, default=env_sensitivity)
    parser.add_argument("--threshold", type=float, default=env_threshold)
    parser.add_argument("--iterations", type=int, default=env_iterations)

    args, unknown = parser.parse_known_args()

    # Compatibility: Check if mode/method were passed as positional arguments
    if not args.mode and len(sys.argv) > 1 and sys.argv[1] in ["attack", "defense"]:
        args.mode = sys.argv[1]
    
    # Validate Required Fields
    if not args.mode:
        print("Error: 'mode' (attack or defense) must be specified via --mode or environment variable 'mode'.")
        parser.print_help()
        sys.exit(1)

    if not args.method:
        print(f"Error: 'method' must be specified via --method or environment variable 'method'.")
        print(f"Supported Attacks: {', '.join(ATTACK_CHARACTERISTICS.keys())}")
        print(f"Supported Defenses: {', '.join(DEFENSE_CHARACTERISTICS.keys())}")
        sys.exit(1)

    # Execute simulation
    if args.mode == "attack":
        # Pass all relevant args to attack simulator
        run_attack_simulation(
            args.method, 
            args.json_dir, 
            args.output_path, 
            input_dir=args.input_path,
            train_subset=args.train_subset,
            test_subset=args.test_subset,
            poison_rate=args.poison_rate,
            trigger_size=args.trigger_size,
            target_label=args.target_label,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    elif args.mode == "defense":
        # Pass all relevant args to defense simulator
        run_defense_simulation(
            args.method, 
            args.json_dir, 
            args.output_path, 
            input_dir=args.input_path,
            train_subset=args.train_subset,
            test_subset=args.test_subset,
            sensitivity=args.sensitivity,
            threshold=args.threshold,
            iterations=args.iterations,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
