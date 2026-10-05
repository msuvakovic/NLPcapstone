"""Resume the complete bounded pilot, then audit and render its results."""
import subprocess
import argparse
import sys
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[2]
    directory=root/'tools'/'prompt_bench'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='output/prompt_benchmark_2026-09-29')
    output=parser.parse_args().output
    steps=[
        ['run.py','--stage','prepare','--output',output],
        ['leakage_check.py'],
        ['audit_old_logs.py'],
        ['run.py','--stage','search','--workers','4','--output',output],
        ['run.py','--stage','evaluate','--workers','4','--output',output],
        ['latency.py','--output',output],
        ['verify.py','--output',output],
        ['analyze.py','--output',output],
        ['plot.py','--output',output],
    ]
    for step in steps:
        print('WORKFLOW STEP',' '.join(step),flush=True)
        subprocess.run([sys.executable,'-X','utf8','-u','-B',str(directory/step[0]),*step[1:]],cwd=root,check=True)
    print('WORKFLOW COMPLETE',flush=True)


if __name__=='__main__': main()
