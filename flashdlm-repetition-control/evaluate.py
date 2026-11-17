"""Evaluate the original model or the complete screenshot method."""
import argparse
from experiment import evaluate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=('original','complete'))
    evaluate(p.parse_args().arm)
