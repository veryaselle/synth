# GitHub upload procedure

The thesis repository used in the manuscript is:

```text
https://github.com/veryaselle/synth
```

## 1. Verify the package before pushing

From the unpacked repository root:

```bash
python scripts/maintenance/verify_clone_ready.py
```

Expected final line: `CLONE-READY CHECK PASS`.

## 2. Initialize or update Git

For a new local checkout:

```bash
git init
git branch -M main
git add .
git status
git commit -m "Prepare clone-ready thesis repository"
git remote add origin https://github.com/veryaselle/synth.git
git push -u origin main
```

If the GitHub repository is already cloned, copy/update these files in that clone and use the normal `git add`, `git commit`, `git push` workflow instead of adding the remote again.

## 3. Environment files

The readable environment specifications are tracked under `environment/`. Exact environment exports may also be committed after reviewing them for credentials and local-prefix fields.

## 4. Large files

Model checkpoints, trainer states, caches and temporary generation chunks are deliberately excluded. If a checkpoint is archived later, use Git LFS rather than committing it as a normal Git object.

```bash
git lfs install
git lfs track "*.safetensors" "*.pt" "*.pth" "*.ckpt" "*.bin"
git add .gitattributes
```

Never commit tokens, credentials, `.env` files or cache directories.

## 5. Freeze the thesis version

After the final repository state has been pushed and re-cloned/verified, create the tag referenced by the thesis:

```bash
git tag -a thesis-v1.0 -m "Code and analysis corresponding to the submitted master's thesis"
git push origin thesis-v1.0
```

On GitHub, create a Release from `thesis-v1.0` with the title `Master's Thesis Reproducibility Release v1.0`.
