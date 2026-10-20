"""Copy only submission artifacts into an isolated official-evaluation directory."""
import json
import shutil
import subprocess
import sys
import zipfile
import hashlib
import re
from pathlib import Path
import nbformat
import joblib

def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'selection_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, f'Frozen input changed: {name}'
    evaluation = root / '.cache/official_evaluation'
    evaluation.mkdir(parents=True, exist_ok=True)
    files = ['train_models.ipynb'] + [f'model_seed{s}.joblib' for s in (42, 2026, 3102)]
    for name in files:
        assert (root / name).exists(), name
        if name.endswith('.joblib'):
            assert (root / name).stat().st_size < 100_000_000
            shutil.copy2(root / name, evaluation / name)
    seed_metadata = {}
    for seed in (42, 2026, 3102):
        model = joblib.load(root / f'model_seed{seed}.joblib')
        states = [member['classifier'].random_state for member in model.members]
        assert len(states) == len(json.loads((root / 'final_config.json').read_text(encoding='utf-8'))['members'])
        assert states == [seed] * len(states), 'Seed metadata does not match filename.'
        seed_metadata[str(seed)] = states
        del model
    source_nb = root / 'Model_Evaluation.ipynb'
    shutil.copy2(source_nb, evaluation / source_nb.name)
    prepared = nbformat.read(evaluation / source_nb.name, as_version=4)
    for cell in prepared.cells:
        if cell.cell_type == 'code':
            cell.execution_count = None
            cell.outputs = []
    nbformat.write(prepared, evaluation / source_nb.name)
    shutil.copytree(root / 'student_data/validation_set_to_students',
                    evaluation / 'validation_set_to_students', dirs_exist_ok=True)
    run = subprocess.run([sys.executable, str(root / 'execute_notebook.py'),
                          str(evaluation / source_nb.name)])
    assert run.returncode == 0, 'Official evaluation process failed.'
    executed = nbformat.read(evaluation / source_nb.name, as_version=4)
    original = nbformat.read(source_nb, as_version=4)
    assert [c.source for c in original.cells] == [c.source for c in executed.cells], 'Official evaluator source changed.'
    code_cells = [cell for cell in executed.cells if cell.cell_type == 'code']
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(output.output_type == 'error' for cell in code_cells for output in cell.outputs)
    assert any('Average test accuracy' in output.get('text', '')
               for cell in code_cells for output in cell.outputs), 'Official evaluation did not finish.'
    summary = json.loads((root / 'results_summary.json').read_text(encoding='utf-8'))
    evaluation_text = '\n'.join(output.get('text', '') for cell in code_cells for output in cell.outputs)
    for row in summary['models']:
        match = re.search(r'model_seed' + str(row['seed']) + r'\.joblib:\s*([0-9.]+)', evaluation_text)
        assert match and abs(float(match.group(1)) - row['validation_accuracy']) <= .00005001
    average = re.search(r'Average test accuracy:\s*([0-9.]+)', evaluation_text)
    assert average and abs(float(average.group(1)) - summary['mean_validation_accuracy']) <= .00005001
    training = nbformat.read(root / 'train_models.ipynb', as_version=4)
    code = ''.join(c.source for c in training.cells if c.cell_type == 'code')
    assert hashlib.sha256(code.encode()).hexdigest() == manifest['notebook_code_sha256']
    assert all(c.execution_count is not None for c in training.cells if c.cell_type == 'code')
    assert not any(o.output_type == 'error' for c in training.cells if c.cell_type == 'code' for o in c.outputs)
    shutil.copy2(evaluation / source_nb.name, root / 'Model_Evaluation_executed.ipynb')
    destination = root / 'submission'
    destination.mkdir(exist_ok=True)
    archive = destination / 'LIU, Yixuan-50027321-AIAA3102-Assignment1.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for name in files:
            z.write(root / name, name)
    with zipfile.ZipFile(archive) as z:
        assert sorted(z.namelist()) == sorted(files)
        assert z.testzip() is None
    verification = dict(official_evaluator_source_unchanged=True,
        frozen_source_and_data_hashes_match=True,
        official_evaluator_all_cells_passed=True, training_run_all_passed=True,
        evaluator_runner_exit_code=run.returncode,
        official_scores_match_training_report=True,
        classifier_random_states=seed_metadata,
        archive=archive.name, archive_bytes=archive.stat().st_size,
        files={name: dict(bytes=(root / name).stat().st_size,
                         sha256=hashlib.sha256((root / name).read_bytes()).hexdigest()) for name in files})
    assert len({verification['files'][f'model_seed{seed}.joblib']['sha256'] for seed in (42,2026,3102)}) == 3
    (root / 'submission_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == '__main__':
    main()
