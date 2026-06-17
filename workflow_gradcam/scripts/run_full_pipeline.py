import subprocess
import sys


def run(cmd):
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    run([sys.executable, "scripts/train_model.py", "--epochs", "15"])
    run([sys.executable, "scripts/generate_cams.py", "--mode", "single", "--output", "outputs/figures/single_gradcam.png"])
    run([sys.executable, "scripts/generate_cams.py", "--mode", "all-classes", "--output", "outputs/figures/all_classes_gradcam.png"])
    run([sys.executable, "scripts/generate_cams.py", "--mode", "one-per-class", "--output", "outputs/figures/one_per_class_gradcam.png"])


if __name__ == "__main__":
    main()
