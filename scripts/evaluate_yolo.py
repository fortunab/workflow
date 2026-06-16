import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--weights", default="models/yolo/best.pt")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data)

    print("\nEvaluation results")
    print("------------------")
    print("mAP@0.5:", float(metrics.box.map50))
    print("mAP@0.5:0.95:", float(metrics.box.map))
    print("Precision:", float(metrics.box.mp))
    print("Recall:", float(metrics.box.mr))

if __name__ == "__main__":
    main()
