import argparse
from utils_reasoning import format_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    args = parser.parse_args()

    instruction = "Generate a clinical reasoning answer from the structured medical-image tokens."
    print(format_prompt(instruction, args.tokens))

if __name__ == "__main__":
    main()
