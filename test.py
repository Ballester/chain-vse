from vocab import Vocabulary
import evaluation
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument('--model', '-m',)
parser.add_argument('--data_path', '-d', default=None)
parser.add_argument('--split', '-s', default='test')
parser.add_argument('--cv', action='store_false')

args = parser.parse_args()

evaluation.evalrank(
    model_path=args.model,
    data_path=args.data_path,
    split=args.split,
    fold5=False,
)
