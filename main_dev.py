import os
import argparse
import tqdm
import torch
import random

import transferattack
from transferattack.utils import *

def get_parser():
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples with incs')
    parser.add_argument('-e', '--eval', action='store_true', help='attack/evluation')
    parser.add_argument('--attack', default='lifgsm', type=str, help='the attack algorithm',
                        choices=['fgsm', 'ifgsm', 'mifgsm', 'nifgsm', 'vmifgsm', 'vnifgsm', 'emifgsm', 'ifgssm', 'vaifgsm', 'aifgtm', 'pcifgsm', 'dta', 'pgn', 'gra', 'iefgsm',
                                'dim', 'tim', 'sim', 'admix', 'dem', 'ssm', 'sia', 'stm', 'bsr',
                                'tap', 'ila', 'fia', 'yaila', 'trap', 'naa', 'rpa', 'taig', 'fmaa', 'ilpd',
                                'sgm', 'dsm', 'mta', 'mup', 'bpa', 'pna_patchout', 'sapr', 'tgr','ngd'
                        ])
    parser.add_argument('--epoch', default=10, type=int, help='the iterations for updating the adversarial patch')
    parser.add_argument('--batchsize', default=10, type=int, help='the bacth size')
    parser.add_argument('--eps', default=16 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--alpha', default=1.6 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--momentum', default=0., type=float, help='the decay factor for momentum based attack')

    parser.add_argument('--white_model', default='tf2torch_inception_v3', type=str, help='the source surrogate model')
    parser.add_argument('--model_dir', type=str, default='./torch_nets_weight/', help='Model weight directory.')
    parser.add_argument('--input_dir', default='./data', type=str, help='the path for custom benign images, default: untargeted attack data')
    parser.add_argument('--output_dir', default='./results', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--helper_folder',default='./helper',type=str, help='the path to store the helper models')
    parser.add_argument('--targeted', action='store_true', help='targeted attack')
    parser.add_argument('--GPU_ID', default='2', type=str)
    return parser.parse_args()

def seed_torch(seed):
    """Set a random seed to ensure that the results are reproducible"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset = AdvDataset(input_dir=args.input_dir, output_dir=args.output_dir, targeted=args.targeted, eval=args.eval)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)
    input_num = len(dataset)
    print("Begin woring on {} images.".format(input_num))
    if not args.eval:
        if args.attack in transferattack.attack_zoo:
            attacker = transferattack.attack_zoo[args.attack.lower()](model_name = args.white_model, model_dir = args.model_dir, targeted = args.targeted)
        else:
            raise Exception("Unspported attack algorithm {}".format(args.attack))

        for batch_idx, [images, labels, filenames] in tqdm.tqdm(enumerate(dataloader)):
            perturbations = attacker(images, labels)
            save_images(args.output_dir, images + perturbations.cpu(), filenames)
    else:
        # Create models
        models = get_models(list_nets, args.model_dir)

        # Initialization parameters
        correct_num = {}
        logits = {}
        for net in list_nets:
            correct_num[net] = 0

        # Start iteration
        for images, labels, _ in dataloader:
            if args.targeted:
                labels = labels[1]
            # Prediction
            with torch.no_grad():
                for net in list_nets:
                    if "inc" in net:
                        logits[net] = models[net](images.cuda())[0]
                    else:
                        logits[net] = models[net](images.cuda())
                    correct_num[net] += (labels.numpy() == logits[net].argmax(dim=1).detach().cpu().numpy()).sum()
        # Print attack success rate
        for net in list_nets:
            if args.targeted:  # correct: pred == target_label #targeted attack
                print('{} attack success rate: {:.2%}'.format(net, correct_num[net] / input_num))
            else:  # correct: pred == original_label
                print('{} attack success rate: {:.2%}'.format(net, 1 - correct_num[net] / input_num))


if __name__ == '__main__':
    seed_torch(0)
    main()

