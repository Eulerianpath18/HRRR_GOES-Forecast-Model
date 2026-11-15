import argparse
import json
from forecastprep.selftest import complete_workflow

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Small synthetic end-to-end check; no scientific forecast results')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    args = parser.parse_args()
    print(json.dumps(complete_workflow(args.output,args.device),indent=2))
