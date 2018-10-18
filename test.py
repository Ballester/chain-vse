from vocab import Vocabulary
import evaluation
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument('--model', '-m',)
parser.add_argument('--data_path', '-d', default=None)
parser.add_argument('--data_name', default=None)
parser.add_argument('--split', '-s', default='test')
parser.add_argument('--no_cv', action='store_false')
parser.add_argument('--batch_size', type=int, default=64)

args = parser.parse_args()

evaluation.evalrank(
    model_path=args.model,
    data_path=args.data_path,
    data_name=args.data_name,
    split=args.split,
    fold5=args.no_cv,
    batch_size=args.batch_size,
)
