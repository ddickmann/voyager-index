# Releasing voyager-index

Step-by-step release checklist for maintainers.

## Standard release

1. **Decide the version** — follow [semver](https://semver.org/):
   - patch (`0.1.1`): bug fix, docs, dependency bump
   - minor (`0.2.0`): new feature, non-breaking API addition
   - major (`1.0.0`): breaking API change

2. **Update `pyproject.toml`**:
   ```toml
   version = "0.X.Y"
   ```

3. **Update `CHANGELOG.md`** — add a new section at the top:
   ```markdown
   ## 0.X.Y

   - description of changes
   ```

4. **Commit**:
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "release: v0.X.Y"
   ```

5. **Push to main**:
   ```bash
   git push origin main
   ```

6. **Wait for CI** — confirm the green checkmark on the commit.

7. **Create the GitHub release**:
   ```bash
   gh release create v0.X.Y --title "0.X.Y" --notes "See CHANGELOG.md for details."
   ```

8. **Trusted publisher auto-publishes to PyPI** — the `release.yml` workflow
   triggers on the published release event and uploads to PyPI via OIDC. This
   typically completes within 2 minutes.

9. **Verify**:
   ```bash
   pip install voyager-index==0.X.Y
   python -c "import voyager_index; print(voyager_index.__version__)"
   ```

## Pre-release (release candidate)

Use a pre-release version tag to test before the final release:

```bash
# In pyproject.toml: version = "0.2.0rc1"
git commit -m "release: v0.2.0rc1"
git push origin main
gh release create v0.2.0rc1 --title "0.2.0rc1" --prerelease --notes "Release candidate."
```

Pre-releases appear on PyPI but are not installed by default (`pip install voyager-index` skips them). Install with:

```bash
pip install voyager-index==0.2.0rc1
```

## Hotfix

1. Fix the issue on `main`.
2. Bump the patch version (e.g., `0.1.0` → `0.1.1`).
3. Follow the standard release steps above.

## Yanking a broken release

If a release has a critical defect:

```bash
# Yank on PyPI (package stays downloadable by exact version, but hidden from resolution)
pip install twine
twine yank voyager-index 0.X.Y

# Or via the PyPI web UI: go to the release page and click "Yank"
```

Then publish a hotfix release immediately.

## Prerequisites

- GitHub CLI (`gh`) authenticated with repo access.
- PyPI trusted publisher configured (see below).

### PyPI trusted publisher setup (one-time)

1. Go to <https://pypi.org/manage/account/publishing/>.
2. Under **Add a new pending publisher**, enter:
   - PyPI project name: `voyager-index`
   - Owner: `ddickmann`
   - Repository: `voyager-index`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. Click **Add**.

After the first successful publish, the pending publisher converts to an active
trusted publisher automatically.
