"""Warm-start both policies immediately using the currently exported teacher data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def run_logged(command, log, *, cwd, env, timeout):
    """Stream child output to the terminal and retain the same text on disk."""
    with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as process:
        def forward():
            for line in process.stdout:
                log.write(line); log.flush()
                print(line, end='', flush=True)
        reader = threading.Thread(target=forward, daemon=True)
        reader.start()
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            process.kill(); process.wait()
            raise
        finally:
            reader.join()
            process.stdout.close()
    return subprocess.CompletedProcess(command, code)


def snapshot(source, destination):
    """Copy only manifest-listed, checksum-verified immutable shards."""
    raw = (source / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    if manifest['schema'] != 'sts2-demonstrations/v1':
        raise ValueError('Unsupported teacher schema')
    signature = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    for entry in manifest['files']:
        path = (source / entry['path']).resolve()
        target = (destination / entry['path']).resolve()
        if not path.is_relative_to(source.resolve()) or not target.is_relative_to(destination.resolve()):
            raise ValueError('Unsafe shard path')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Teacher checksum mismatch')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (destination / 'manifest.json').write_bytes(raw)
    return signature


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT.parent / 'sts2-ml/runs/fireworks-full-generation-20261101/dataset')
    p.add_argument('--fresh', action='store_true', help='Start new policies instead of continuing saved checkpoints')
    p.add_argument('--output', type=Path, default=ROOT / 'output/teachers_overnight_v3')
    p.add_argument('--hours', type=float, default=10, help='Maximum total training time')
    p.add_argument('--epochs', type=int, default=20, help='Additional epochs per policy on this invocation')
    a = p.parse_args()
    if a.hours <= 0 or a.epochs < 1:
        p.error('hours and epochs must be positive')
    a.data = a.data.resolve(); a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    lock = a.output / 'runner.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    state_path = a.output / 'status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'roles': {}}
    state.update(status='running', pid=os.getpid(), data=str(a.data), hours=a.hours)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def save(status):
        state.update(status=status, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        temp = state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2)); temp.replace(state_path)

    try:
        save('preparing_snapshot')
        print(f'Preparing current teacher snapshot: {a.data}', flush=True)
        deadline = time.monotonic() + a.hours * 3600
        if (a.output / 'STOP').exists():
            save('operator_stopped'); return
        signature = snapshot(a.data, a.output / 'snapshot')
        state.pop('waiting_reason', None)
        for role in ('combat', 'strategy'):
            if time.monotonic() >= deadline: break
            previous = state['roles'].get(role, {})
            attempt=1
            while (a.output / role / f'pass-{attempt:04d}').exists(): attempt+=1
            folder = a.output / role / f'pass-{attempt:04d}'
            folder.mkdir(parents=True, exist_ok=True)
            output = folder / 'model'
            command = [sys.executable, '-u', str(ROOT / 'scripts/pretrain_demonstrations.py'),
                       '--data', str(a.output / 'snapshot'), '--role', role,
                       '--output', str(output), '--epochs', str(a.epochs), '--observation-version', '3']
            if previous.get('checkpoint') and not a.fresh:
                print(f"Continuing {role} from {previous['checkpoint']}",flush=True)
                command += ['--initial', previous['checkpoint']]
            state['active_role'] = role; save('training')
            print(f'Training {role}: {signature[:16]}', flush=True)
            with (folder / 'training.log').open('w') as log:
                try:
                    result = run_logged(command, log, cwd=ROOT, env=env,
                                        timeout=max(.01, deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    print('Training time limit reached; stopping the active trainer.', flush=True)
                    save('time_limit_reached'); return
            if result.returncode:
                raise RuntimeError(f'{role} training failed; see {folder / "training.log"}')
            report = json.loads(output.with_suffix('.report.json').read_text())
            state['roles'][role] = {'signature': signature, 'report': report,
                'total_epochs': (0 if a.fresh else previous.get('total_epochs',0)) + a.epochs,
                'checkpoint': str(output.with_suffix('.zip')) if report['train_examples'] else previous.get('checkpoint')}
            save('running')
            print(f"Finished {role}: {state['roles'][role]['checkpoint']}", flush=True)
        state.pop('active_role', None)
        if time.monotonic() >= deadline:
            save('time_limit_reached')
        else:
            save('complete')
        print(f"Finished. Checkpoints and reports: {a.output}", flush=True)
        return
    except KeyboardInterrupt:
        save('operator_stopped')
    except Exception as error:
        state['error'] = str(error); save('failed'); raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
