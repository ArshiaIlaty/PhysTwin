# Lessons learned

(Append a dated entry every time the user corrects an approach, OR confirms a
non-obvious decision worked. Lead with the rule, then **Why** and **How to apply**.)

## 2026-05-20 — Confirm critical assumptions before bulk work
- **Rule**: When a multi-day plan has architectural ambiguities (e.g. variable
  feature dims, force semantics), surface them and lock decisions with the user
  *before* doing irreversible setup work (downloads, env builds).
- **Why**: Original plan in `experiment_plan.md` had a hidden bug
  (`np.concatenate` of features with varying `N_particles`) that would have
  surfaced mid-Stage 3 after ~10h of work.
- **How to apply**: For non-trivial implementation requests, do a critical
  read-through and AskUserQuestion on architectural forks first.

## 2026-05-20 — SLURM defaults on HPC3 for this project
- **Rule**: Every GPU job uses `--partition=free-gpu --account=mgamalel`.
- **Why**: User explicitly directed this account/partition combo.
- **How to apply**: Bake into every `sbatch`/`srun` invocation, including
  interactive sessions.

## 2026-05-20 — pip on beegfs (/pub) can hang in IB recv
- **Rule**: When `pip install` to `/pub/mgamalel/envs/...` is stuck in `D`
  state with `wchan=IBVSocket_waitForRecvCompletionEvent` for several minutes,
  it eventually unblocks (saw ~11 min stall today). Don't panic-rebuild the env
  on /share/crsp NFS unless the wait exceeds ~20 min.
- **Why**: beegfs over Infiniband has occasional sustained recv stalls; pip's
  metadata writes (esp. of dist-info) trigger them. The install resumes
  cleanly once IB recovers.
- **How to apply**: For >20 min stalls, fallback location is
  `/share/crsp/lab/selmalak/mgamalel/envs/`. For shorter stalls, just wait;
  killing the bash parent is safer than `kill -9` the stuck pip (which can't
  be reaped while in `D` state).

## 2026-05-20 — Compile CUDA submodules on CPU node needs `TORCH_CUDA_ARCH_LIST`
- **Rule**: When building torch C++/CUDA extensions (diff-gaussian-rasterization,
  simple-knn) on a CPU node, you MUST `export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9"`
  (covers HPC3's V100/A30/A100/A40/A6000/L40S). Without it, torch raises
  `IndexError: list index out of range` from `_get_cuda_arch_flags()`.
- **Why**: torch tries to detect GPU arch via `torch.cuda.get_arch_list()` on
  the local machine; if there's no GPU it returns empty and the build fails.
- **How to apply**: Add `export TORCH_CUDA_ARCH_LIST=...` to every build step
  for compiled CUDA extensions in this project.

## 2026-05-20 — pip submodule builds: install from /tmp, not /share/crsp
- **Rule**: For `pip install --no-build-isolation ./gaussian_splatting/submodules/...`,
  copy the submodule to `/tmp/...` first and install from there. Builds from
  /share/crsp (NFS) hit "Errno 5 Input/output error" writing SOURCES.txt.
- **Why**: pip's metadata writes during install fail intermittently on NFS
  but work fine on the local /tmp scratch disk.
- **How to apply**: Wrap the submodule install in: `cp -r src /tmp/x && pip
  install --no-build-isolation /tmp/x && rm -rf /tmp/x`.

## 2026-05-20 — `unzip` on HF data.zip needs `UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE`
- **Rule**: PhysTwin's `data.zip` (~8GB) trips newer unzip's zip-bomb heuristic
  (overlapped components). Set `UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE` before
  the `unzip` call.
- **Why**: data.zip has densely packed components by design; unzip 6+ refuses
  to extract it without the override.
- **How to apply**: Bake the env var into any unzip helper script.

## 2026-05-20 — Use parallel wget per file for HuggingFace data
- **Rule**: Sequential single-stream wget from huggingface.co can stall mid-
  file. Run one wget per file in parallel (4 PIDs) so a single stall doesn't
  block the others.
- **Why**: We saw a sequential download stall on data.zip for several minutes
  with TCP connection alive. Parallel streams sustained higher throughput
  AND finished the other 3 files in minutes while the stalled one resumed.
- **How to apply**: Always parallelize HF dataset downloads; use
  `wget --continue --tries=20 --retry-connrefused --waitretry=10 --timeout=60`.
