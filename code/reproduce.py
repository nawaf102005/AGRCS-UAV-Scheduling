"""One entry point for a smoke check or complete independent reproduction."""
import argparse,subprocess,sys,os
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--profile',choices=['quick','paper'],default='quick')
p.add_argument('--retrain',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1];code=root/'code';out=root/'results'/('smoke' if a.profile=='quick' else 'reproduced')
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
def run(script,*args):subprocess.run([sys.executable,str(code/script),*map(str,args)],check=True,env=env)
run('verify.py')
if a.retrain:run('train_ddqn.py','--steps',20000,'--seeds',3)
run('run_experiments.py','--profile',a.profile,'--out',out)
if a.profile=='paper':
    run('run_bulk.py','--out',out)
    run('benchmark_latency.py','--out',out)
    run('build_report.py','--results',out)
    run('review_additions.py','--results',out)
    # Additional review runs retain a separate results/reviewer directory.
    run('reviewer_experiments.py')
    run('union_stress_extension.py')
    run('reviewer_report.py','--baseline-results',out)
    run('draw_system_model.py')
print('Reproduction complete:',out)
