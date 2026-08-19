# Synthetic end-to-end fixture

No recorded speech or third-party ATC audio is stored here. Generate four deterministic, speech-like reference WAVs and run the checked-in fixture configuration with:

```bash
uv run python scripts/generate_fixture_audio.py tests/.artifacts/reference
uv run atc-benchmark build --config tests/fixtures/showcase_test.yaml --force
uv run atc-benchmark validate --output tests/.artifacts/output/showcase_test
```

The outputs live under the gitignored `tests/.artifacts/` directory. The main end-to-end pytest uses an isolated temporary directory and expands the production configuration to all 25 scenarios.
