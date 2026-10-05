# Releasing (maintainer notes)

## This node (`monarchrt`)

`.github/workflows/publish.yml` publishes to the Comfy Registry when
`pyproject.toml` changes on `main` (or by hand). It validates and packs the
node with `comfy-cli`, waits for the regular CI of the same commit, refuses a
version that already exists, and passes the `REGISTRY_ACCESS_TOKEN` secret
only to the official publish action. Bump `[project].version` for every
release; published versions and GitHub release tags are never replaced.

## Why `publish-causalforcing.yml` is in this repository

[ComfyUI-CausalForcing](https://github.com/hiroki-abe-58/ComfyUI-CausalForcing)
belongs to the same registry publisher. The publisher token is stored only as
a secret of this repository; rather than copying it to a second repository,
`publish-causalforcing.yml` publishes that node from here:

- it runs only by hand (`workflow_dispatch`) from this repository's `main`;
- the target repository (`hiroki-abe-58/ComfyUI-CausalForcing`) and node id
  (`causalforcing`) are fixed in the workflow; the only inputs are a full
  commit SHA and a version;
- the first job, which never sees the token, checks that the target
  repository is public, that the commit is on its `main` branch, that tag
  `v<version>` points to exactly that commit and has a GitHub release, that
  the target's CI succeeded for it, that the version is not in the registry
  yet, and that `pyproject.toml` names that node, version, publisher and
  repository; it then runs `comfy node validate` and `comfy node pack` on the
  target and inspects the archive (no tests, scripts, weights, MonarchRT
  files, private paths or token-like strings);
- the second job checks out the same commit again into a fresh runner, runs
  none of its scripts, and gives the token only to the official
  `Comfy-Org/publish-node-action` (pinned by commit), which packs and uploads
  that checkout;
- both publish workflows share one concurrency group, so two registry uploads
  never run at the same time.

To publish a CausalForcing release: make sure its release tag and CI are in
place, then run *Actions -> Publish ComfyUI-CausalForcing to Comfy Registry ->
Run workflow* on `main` with the tag's commit SHA and the version.

## `publish-leaptalk.yml` (ComfyUI-LeapTalk)

[ComfyUI-LeapTalk](https://github.com/hiroki-abe-58/ComfyUI-LeapTalk) is published from here the same
way: dispatch only, from `main`; target repository (`hiroki-abe-58/ComfyUI-LeapTalk`) and node id
(`leaptalk`) fixed; inputs are a full commit SHA and a version; the first job (no token) checks
visibility, `main`, tag, release, CI, that the version is new, `pyproject.toml` and the packed archive
(which must contain the persistent-worker files of LeapTalk 0.2.0 and later).

Difference from the CausalForcing workflow: the publish job does not use
`Comfy-Org/publish-node-action`, because that action installs `comfy-cli` without a version and uses a
floating `setup-python` tag inside a step that receives the token. It runs the same command
(`comfy node publish`) with `comfy-cli==1.22.0` (pinned, reviewed) and `setup-python` pinned by commit,
re-checks the commit, `pyproject.toml` and that the version is new right before publishing, disables
comfy-cli telemetry, and passes the token only to that last command.

To publish a LeapTalk release: make sure its release tag and CI are in place, then run *Actions ->
Publish ComfyUI-LeapTalk to Comfy Registry -> Run workflow* on `main` with the tag's commit SHA and the
version.

## `publish-tbdub.yml` (ComfyUI-TBDub)

[ComfyUI-TBDub](https://github.com/hiroki-abe-58/ComfyUI-TBDub) (node id `tbdub`) is published from here with a
copy of `publish-leaptalk.yml`: same guards, same pinned `comfy-cli==1.22.0`, token only in the last step; the
target repository and node id are fixed to TBDub, and the package check requires the TBDub runtime files and
refuses tests, tools, measurement records and media.

To publish a TBDub release: make sure its release tag and CI are in place, then run *Actions -> Publish
ComfyUI-TBDub to Comfy Registry -> Run workflow* on `main` with the tag's commit SHA and the version.

## `publish-pixrestore.yml` (ComfyUI-PixRestore)

[ComfyUI-PixRestore](https://github.com/hiroki-abe-58/ComfyUI-PixRestore) (node id `pixrestore`) is published from
here with a copy of `publish-looped-dit.yml`: same guards, same pinned `comfy-cli==1.22.0`, token only in the last
step; the target repository and node id are fixed to PixRestore, and the package check requires the PixRestore
runtime files and refuses tests, scripts, upstream `pixrestore/` or `dinov2/` code, weights and media (except the one
sample input under `workflows/input/`).

To publish a PixRestore release: make sure its release tag and CI are in place, then run *Actions -> Publish
ComfyUI-PixRestore to Comfy Registry -> Run workflow* on `main` with the tag's commit SHA and the version.

## `publish-miripple.yml` (ComfyUI-MiRipple)

[ComfyUI-MiRipple](https://github.com/hiroki-abe-58/ComfyUI-MiRipple) (node id `mi-ripple`) is published from here
with a copy of `publish-pixrestore.yml`: same guards, same pinned `comfy-cli==1.22.0`, token only in the last step;
the target repository and node id are fixed to MiRipple, and the package check requires the node files and the
vendored MIT Mi-Ripple code with its license copy (each vendored file must match `VENDOR.json`), and refuses tests,
scripts, docs images, media and weights.

To publish a MiRipple release: make sure its release tag and CI are in place, then run *Actions -> Publish
ComfyUI-MiRipple to Comfy Registry -> Run workflow* on `main` with the tag's commit SHA and the version.