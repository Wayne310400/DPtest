import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output', type=str, default='epsilon', help="output type")
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--epoch', type=int, default=100, help="rounds of training")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--dp_mechanism', type=str, default='gaussian',
                        help='differential privacy mechanism')
    parser.add_argument('--epsilon', type=float, default=10.0, help='single epsilon')
    parser.add_argument('--epsilons', type=float, nargs='+',
                        help="epsilon of array")
    parser.add_argument('--dp_ratios', type=float, nargs='+', 
                        help='differential privacy noise ratio of array')
    parser.add_argument('--dp_ratio', type=float, 
                        help='single differential privacy noise ratio')

    args = parser.parse_args()
    return args