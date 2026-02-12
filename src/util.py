import json
from torch.utils.tensorboard import SummaryWriter

def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))

def start_writer(simulator_params, model_params, model_name):
    writer = SummaryWriter(f'./runs/{model_name}')
    model_params['checkpoint_path'] = f'{writer.log_dir}/checkpoint.pt'
    writer.add_text('Simulator parameters', pretty_json(simulator_params))
    writer.add_text('Model parameters', pretty_json(model_params))
    writer.flush()
    return writer