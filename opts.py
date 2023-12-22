import argparse


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_path', type=str, default='/data/sh/Cost-Free Backdoor attacks/data_path')
    parser.add_argument('--data_name', type=str, default='c10', choices=['c10', 'c100', 'i200'])
    parser.add_argument('--model_name', type=str, default='v16', choices=['v16', 'r18', 'p18'])
    parser.add_argument('--attack_name', type=str, default='blend', choices=['blend', 'bad', 'opti'])
    parser.add_argument('--select_name', type=str, default='rand', choices=['rand', 'pfs', 'fus', 'pf'])

    parser.add_argument('--result_path', type=str, default='./results_save/results')
    parser.add_argument('--sample_idx_path', type=str, default='./results_save/sample_idxs')
    parser.add_argument('--print_path', type=str, default='./results_save/print_txt')

    parser.add_argument('--attack_target', type=int, default=0)
    parser.add_argument('--transform_id', type=int, default=0, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument('--fus_n_iter', type=int, default=15)
    parser.add_argument('--suffix', type=int, default=0)
    parser.add_argument('--m', type=int, default=10)

    parser.add_argument('--poison_ratio', type=float, default=0.01)
    parser.add_argument('--fus_alpha', type=float, default=0.7)

    opts = parser.parse_args()
    return opts
