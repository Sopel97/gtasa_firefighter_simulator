import sys
import subprocess
from multiprocessing.dummy import Pool

def make_on_success_callback(command):
    def callback(result):
        print(f'INFO: Completed command: `{command}`')
    return callback

def make_on_failure_callback(command):
    def callback(result):
        print(f'ERROR: Failed command: `{command}`')
    return callback

def run_command(command):
    print(f'INFO: Executing command: `{command}`')
    subprocess.run(f'{sys.executable} main.py {command}', stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, universal_newlines=True)

if __name__ == '__main__':
    commands_filename = sys.argv[1]
    concurrency = int(sys.argv[2])

    with Pool(concurrency) as pool:
        results = []
        with open(commands_filename, 'r') as commands_file:
            for line in commands_file:
                command = line.strip()
                if command and not command.startswith('#'):
                    results.append(pool.apply_async(run_command, [command], callback=make_on_success_callback(command), error_callback=make_on_failure_callback(command)))

        for result in results:
            result.wait()
