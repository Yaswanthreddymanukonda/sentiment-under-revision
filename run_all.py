# Run the pipeline: experiments → figures → check against the archived numbers.

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    env = os.environ.copy()
    # pin BLAS to 1 thread, otherwise the runtime jumps around a lot
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['OMP_NUM_THREADS'] = '1'

    for script in ['run_experiments.py', 'make_figures.py', 'verify_results.py']:
        print(f'\nRunning {script}', flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / script)],
            cwd=ROOT,
            env=env,
            check=True,
        )


if __name__ == '__main__':
    main()
