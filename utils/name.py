

def get_name(opts):
    name = 'error'

    if opts.select_name == 'rand' or opts.select_name == 'pfs':
        name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(
            opts.data_name,
            opts.attack_name,
            opts.select_name,
            opts.model_name,
            opts.poison_ratio,
            opts.attack_target,
            opts.transform_id,
            opts.suffix,
        )

    if opts.select_name == 'fus' or opts.select_name == 'pf' or opts.select_name == 'transfer':
        name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            opts.data_name,
            opts.attack_name,
            opts.select_name,
            opts.model_name,
            opts.poison_ratio,
            opts.fus_alpha,
            opts.fus_n_iter,
            opts.attack_target,
            opts.transform_id,
            opts.suffix,
        )


    return name
