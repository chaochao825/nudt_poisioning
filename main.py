import argparse
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from attack_simulator import run_attack_simulation, ATTACK_CHARACTERISTICS
from defense_simulator import run_defense_simulation, DEFENSE_CHARACTERISTICS

def main():
    # Priority: Command line args > Environment variables > Defaults
    env_mode = os.getenv("mode")
    env_method = os.getenv("method")
    env_json_dir = os.getenv("json_dir", "./json_file")
    env_output_path = os.getenv("OUTPUT_PATH", "./output")
    env_input_path = os.getenv("INPUT_PATH", "./input")

    parser = argparse.ArgumentParser(description="NUDT Poisoning Attack & Defense System")
    
    # Required Arguments (can be set via env vars)
    parser.add_argument("--mode", type=str, choices=["attack", "defense"], default=env_mode, 
                        help="REQUIRED: Execution mode. Supported: attack, defense")
    parser.add_argument("--method", type=str, default=env_method, 
                        help="REQUIRED: Method name. \n"
                             "Supported Attacks: BadNets, Trojan, Feature Collision, Triggerless, Dynamic Backdoor, Physical Backdoor, Neuron Interference, Model Poisoning. \n"
                             "Supported Defenses: STRIP, NC, DifferentialPrivacy.")

    # General Optional Arguments
    parser.add_argument("--json_dir", type=str, default=env_json_dir, help="Directory containing JSON templates")
    parser.add_argument("--output_path", type=str, default=env_output_path, help="Directory for output reports")
    parser.add_argument("--input_path", type=str, default=env_input_path, help="Directory for input datasets")

    # Attack-Specific Optional Arguments
    parser.add_argument("--poison_rate", type=float, default=0.1, help="Poisoning ratio (0.0 to 1.0)")
    parser.add_argument("--trigger_size", type=int, default=3, help="Size of the trigger in pixels")
    parser.add_argument("--target_label", type=int, default=0, help="Target class label for the attack")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimization")

    # Defense-Specific Optional Arguments
    parser.add_argument("--sensitivity", type=float, default=0.5, help="Defense sensitivity (0.0 to 1.0)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations for defense analysis")

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
            sensitivity=args.sensitivity,
            threshold=args.threshold,
            iterations=args.iterations
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
