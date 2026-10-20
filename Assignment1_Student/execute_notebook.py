"""Execute every notebook cell in a fresh kernel and retain actual outputs."""
import os
import sys
import json
import tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
os.environ['JUPYTER_PATH'] = str(root / '.cache/jupyter')
os.environ['IPYTHONDIR'] = str(root / '.cache/ipython')
os.environ['JUPYTER_RUNTIME_DIR'] = str(root / '.cache/jupyter-runtime')
os.environ['LOKY_MAX_CPU_COUNT'] = '6'
os.environ['OMP_NUM_THREADS'] = '6'
os.environ['OPENBLAS_NUM_THREADS'] = '6'
runtime = root / '.cache/jupyter-runtime'
runtime.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(runtime)
os.environ['TEMP'] = str(runtime)
os.environ['TMP'] = str(runtime)
kernel = root / '.cache/jupyter/kernels/assignment1'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps(dict(
    argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    display_name='Assignment 1 Python 3.14.6', language='python')), encoding='utf-8')
path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root / 'train_models.ipynb'
notebook = nbformat.read(path, as_version=4)
def finished(cell, cell_index, **kwargs):
    nbformat.write(notebook, path)
    print(f'Completed cell {cell_index}', flush=True)
    for output in cell.get('outputs', []):
        if output.output_type == 'stream':
            print(output.text[-1800:], flush=True)
class StreamingNotebookClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg.get('msg_type') == 'stream':
            print(msg['content']['text'], end='', flush=True)
        return super().process_message(msg, cell, cell_index)

client = StreamingNotebookClient(notebook, timeout=14400, kernel_name='assignment1',
    resources={'metadata': {'path': str(path.parent)}}, on_cell_executed=finished)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
print('RUN ALL PASSED:', path, flush=True)
# With this embedded Windows runtime, native Jupyter/ZMQ cleanup at interpreter
# teardown returned STATUS_INVALID_HANDLE after a successful kernel shutdown.
# Only reach this exit after execute() has succeeded and the notebook is saved;
# execution exceptions above still propagate as failures.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
