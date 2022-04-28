import yaml
from itertools import product
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    "--num_seeds",
    type=int,
    default=1
)

args = parser.parse_args()

template = "conda activate harl_modified && python src/main.py --config=qmix --env-config=overcooked with agent={0} env_args.env_config.mdp_params.layout_name={1} seed={2}"

config = {"session_name": "run-all", "windows": []}

agent_list = ["rnn", "mlp"]
layout_list = ["forced_coordination", "coordination_ring"]
seed_list = []

for i in range(args.num_seeds):
    panes_list = []
    for a, l in product(agent_list, layout_list):
        panes_list.append(
            template.format(a, l, i))

    config["windows"].append({
        "window_name": "seed-{}".format(i),
        "panes": panes_list
    })

yaml.dump(config, open("run_all.yaml", "w"), default_flow_style=False)