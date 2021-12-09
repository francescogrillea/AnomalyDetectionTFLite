from telemanom.detectorLite import DetectorLite
import argparse

parser = argparse.ArgumentParser(description='Parse path to anomaly labels if provided.')
parser.add_argument('-l', '--labels_path', default=None, required=False)
args = parser.parse_args()

if __name__ == '__main__':
    detector = DetectorLite(labels_path=args.labels_path)
    detector.run()