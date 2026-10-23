"""Capture approved install/canary processes with disk and cumulative run caps."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SESSION = HERE.parents[3]
TEMP = SESSION / 'canary-install-temp'
ENV = SESSION / 'canary-env'
CAP = 1024**3


def allocated(paths):
    names = [str(path) for path in paths if path.exists()]
    if not names:
        return 0
    out = subprocess.check_output(['du', '-s', '-B1', *names], text=True)
    return sum(int(line.split()[0]) for line in out.splitlines())


def process_cpu(pid):
    try:
        fields = Path('/proc/' + str(pid) + '/stat').read_text().rsplit(')', 1)[1].split()
        return sum(int(fields[i]) for i in (11, 12, 13, 14)) / os.sysconf('SC_CLK_TCK')
    except FileNotFoundError:
        return 0


def cpu(usage):
    return usage.ru_utime + usage.ru_stime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('dependency-download', 'install', 'numerical'))
    parser.add_argument('label')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or '/' in args.label or args.label in ('', '.', '..'):
        raise ValueError('A command and simple unique label are required')
    output = HERE / 'execution'
    output.mkdir(exist_ok=True)
    files = {name: output / (args.label + '.' + suffix) for name, suffix in
             [('stdout', 'stdout'), ('stderr', 'stderr'), ('resource', 'resource.json')]}
    if any(path.exists() for path in files.values()):
        raise RuntimeError('Refusing to overwrite a previous process receipt')
    protocol = json.loads((HERE / 'CANARY_PROTOCOL.json').read_text())
    if not protocol['authorization']['approved']:
        raise RuntimeError('Exact donor authorization is required')
    numerical = args.phase == 'numerical'
    previous = [json.loads(path.read_text()) for path in output.glob('*.resource.json')]
    previous = [row for row in previous if row['phase'] == 'numerical']
    spent_cpu = sum(row['charged_cpu_seconds'] for row in previous)
    spent_wall = sum(row['charged_wall_seconds'] for row in previous)
    remaining_cpu = math.floor(600 - spent_cpu - 3)
    limit_wall = 900 - spent_wall - 2 if numerical else 300
    if numerical and (remaining_cpu < 1 or limit_wall < 1):
        raise RuntimeError('Cumulative numerical allowance exhausted')
    roots = [ENV, TEMP, HERE]
    peak = allocated(roots)
    if peak >= CAP:
        raise RuntimeError('Additional environment/temp/results disk cap already reached')
    environment = os.environ.copy()
    environment.update(TMPDIR=str(TEMP), PYTHONDONTWRITEBYTECODE='1',
                       XDG_CACHE_HOME=str(TEMP / 'cache'), TORCH_HOME=str(TEMP / 'torch'),
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       CUDA_VISIBLE_DEVICES='')
    if numerical:
        environment['KRON_CANARY_CPU_SECONDS'] = str(remaining_cpu)
    start = time.monotonic()
    own_start = cpu(resource.getrusage(resource.RUSAGE_SELF))
    children_start = cpu(resource.getrusage(resource.RUSAGE_CHILDREN))
    stopped = None
    with files['stdout'].open('xb') as stdout, files['stderr'].open('xb') as stderr:
        child = subprocess.Popen(command, cwd=HERE, env=environment, stdout=stdout, stderr=stderr)
        while child.poll() is None:
            used = time.monotonic() - start
            peak = max(peak, allocated(roots))
            artifacts = allocated([HERE])
            measured_cpu = (cpu(resource.getrusage(resource.RUSAGE_SELF)) - own_start
                            + cpu(resource.getrusage(resource.RUSAGE_CHILDREN)) - children_start
                            + process_cpu(child.pid))
            if peak >= CAP:
                stopped = '1 GiB environment/temp/results cap'
            elif artifacts >= 50 * 1024**2:
                stopped = '50 MiB result-artifact cap'
            elif used >= limit_wall:
                stopped = 'cumulative numerical wall cap' if numerical else 'install-stage wall cap'
            elif numerical and spent_cpu + measured_cpu >= 598:
                stopped = 'cumulative numerical CPU cap'
            if stopped:
                child.terminate()
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                break
            try:
                child.wait(timeout=min(1, max(.001, limit_wall - used)))
            except subprocess.TimeoutExpired:
                pass
    used_cpu = (cpu(resource.getrusage(resource.RUSAGE_SELF)) - own_start
                + cpu(resource.getrusage(resource.RUSAGE_CHILDREN)) - children_start)
    peak = max(peak, allocated(roots))
    record = {'phase': args.phase, 'label': args.label, 'command': command,
              'exit_code': child.returncode, 'stopped_by_guard': stopped,
              'charged_cpu_seconds': used_cpu, 'charged_wall_seconds': time.monotonic() - start,
              'numerical_cpu_before': spent_cpu, 'numerical_wall_before': spent_wall,
              'numerical_process_cpu_limit': remaining_cpu if numerical else None,
              'peak_sampled_environment_temp_results_bytes': peak,
              'disk_cap_bytes': CAP, 'disk_sample_interval_seconds': 1,
              'preflight_required': 'Wheel expansion sizes are checked before offline install; disk sampling is additional enforcement.',
              'maximum_child_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
              'stdout_bytes': files['stdout'].stat().st_size,
              'stderr_bytes': files['stderr'].stat().st_size}
    files['resource'].write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps(record, sort_keys=True))
    return 1 if stopped else (child.returncode if child.returncode >= 0 else 128 - child.returncode)


if __name__ == '__main__':
    sys.exit(main())
