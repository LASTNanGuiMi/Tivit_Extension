# Anonymous artifact checklist

Run the following before preparing a review artifact:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neurosigvit
cd NeuroSigViT-main
python scripts/test_anonymity.py
python scripts/check_anonymity.py
```

The runtime writes sanitized arguments: absolute machine paths are replaced by
`<ABSOLUTE_PATH>/<basename>` in `args.json` and feature-cache signatures. The
default experiment wrappers pass relative paths, so dry-run commands are also
machine-independent.

Do not include local datasets, symbolic links used to reach them, results,
feature caches, checkpoints, logs, environment files, or Git metadata in the
review archive. Git metadata contains development history and local repository
configuration even when the checked-out source is anonymous. Use a clean export
of the reviewed source files rather than compressing the working directory.

Third-party dataset links, model identifiers, citations, and license attributions
must remain intact; they identify dependencies and prior work, not the submission
authors.
