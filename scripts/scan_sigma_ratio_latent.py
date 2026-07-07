#!/usr/bin/env python3
"""Generate samples with different latent draw std (sigma) and evaluate classifier accuracy
for different real/synthetic ratios. Saves CSV `results/sigma_ratio_grid_latent_split5.csv`.
"""
import subprocess
import argparse
from pathlib import Path
import json
import time
import pandas as pd


def run_cmd(cmd, capture=False, env=None, timeout=None):
    print('RUN:', ' '.join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    out = res.stdout.decode('utf-8') if capture and res.stdout is not None else ''
    return res.returncode, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sigmas', default='0.5,1.0,1.5,2.0')
    p.add_argument('--ratios', default='1.0,0.8,0.6,0.4,0.2,0.0')
    p.add_argument('--samples-per-class', type=int, default=200)
    p.add_argument('--ddim-steps', type=int, default=50)
    p.add_argument('--model-path', default='models/diffusion_normalized_beta0.20_temp_split_5_best.pt')
    p.add_argument('--decoder-path', default='models/decoder_split_5.pth')
    p.add_argument('--out-csv', default='results/sigma_ratio_grid_latent_split5.csv')
    p.add_argument('--device', choices=['cpu','auto'], default='cpu')
    args = p.parse_args()

    sigmas = [float(x) for x in args.sigmas.split(',') if x.strip()]
    ratios = [float(x) for x in args.ratios.split(',') if x.strip()]

    results = []

    env = None
    if args.device == 'cpu':
        env = dict(**subprocess.os.environ)
        env['CUDA_VISIBLE_DEVICES'] = ''

    for sigma in sigmas:
        prefix = f'grid_s5_sigma_{sigma:.2f}'.replace('.', 'p')
        # generate
        gen_cmd = [
            'python3', 'scripts/generate_and_decode_full.py',
            '--model-path', args.model_path,
            '--decoder-path', args.decoder_path,
            '--output-prefix', prefix,
            '--samples-per-class', str(args.samples_per_class),
            '--ddim-steps', str(args.ddim_steps),
            '--sigma', str(sigma),
            '--sigma-mode', 'fixed'
        ]
        code, out = run_cmd(gen_cmd, capture=True, env=env, timeout=1200)
        if code != 0:
            print('Generation failed for sigma', sigma)
            print(out)
            continue

        spectra_path = f'results/{prefix}_spectra.npy'
        labels_path = f'results/{prefix}_labels.npy'

        for ratio in ratios:
            csv_out = f'results/sigma_ratio_grid_latent_s{sigma:.2f}_r{ratio:.2f}.csv'.replace('.', 'p')
            eval_cmd = [
                'python3', 'scripts/train_mlp_and_eval.py',
                '--real-feather', 'Data/train_data.feather',
                '--test-feather', 'Data/test_data.feather',
                '--synthetic-spectra', spectra_path,
                '--synthetic-labels', labels_path,
                '--ratio', str(ratio),
                '--save-model', f'models/tmp_mlp_s{sigma:.2f}_r{ratio:.2f}.joblib'.replace('.', 'p'),
                '--out-csv', csv_out,
                '--mlp-max-iter', '200'
            ]
            code2, out2 = run_cmd(eval_cmd, capture=True, env=env, timeout=600)
            if code2 != 0:
                print('Evaluation failed for sigma', sigma, 'ratio', ratio)
                print(out2)
                continue
            # read csv
            try:
                df = pd.read_csv(csv_out)
                acc = float(df['accuracy'].iloc[0])
            except Exception as e:
                print('Failed to read', csv_out, e)
                acc = None
            results.append({'sigma': sigma, 'ratio': ratio, 'accuracy': acc})
            # small sleep to avoid hammering
            time.sleep(1.0)

    out_df = pd.DataFrame(results)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print('Wrote', args.out_csv)


if __name__ == '__main__':
    main()
