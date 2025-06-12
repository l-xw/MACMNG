import argparse


parser = argparse.ArgumentParser(description='Arguments for Distributed_Fuzzy_FJSP')
parser.add_argument('--device', type=str, default="cuda", help='devices')

parser.add_argument('--factor_number', type=int, default=1, help='Number of factories')
parser.add_argument('--job_number', type=int, default=5, help='Number of jobs')
parser.add_argument('--operation_number', type=int, default=5, help='Number of operations')
parser.add_argument('--machine_number', type=int, default=4, help='The number of machines in each factory')
parser.add_argument('--threshold_time', type=int, default=1, help='The lower limit of the random fuzzy processing time')
parser.add_argument('--upper_limit_time', type=int, default=15, help='The upper limit of random fuzzy processing time')
parser.add_argument('--unit_energy_consumption', type=int, default=3, help='PP')
parser.add_argument('--seed', type=int, default=0, help='Random number seed')

parser.add_argument('--floors_number', type=int, default=2, help='Number of hidden layers')
parser.add_argument('--n_hidden', type=int, default=512, help='Number of neurons in the hidden layer')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
parser.add_argument('--epsilon', type=float, default=0.9, help='Exploration rate (1 - Exploration rate)')
parser.add_argument('--target_update', type=int, default=10, help='Target network update frequency')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

parser.add_argument('--capacity', type=int, default=200, help='Experience replays buffer size')
parser.add_argument('--min_capacity', type=int, default=50, help='Minimum sample size')

history_data = []
configs = parser.parse_args()


def up_configs(factor_number, job_number, operation_number, machine_number):
    configs.factor_number = factor_number
    configs.job_number = job_number
    configs.operation_number = operation_number
    configs.machine_number = machine_number
