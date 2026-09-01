"""Integration tests for Click CLI commands in main.py."""

import json
import os
import tempfile
from click.testing import CliRunner

from main import cli


def test_cli_help():
    """Validates that all CLI commands display help information."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Face identification, genuine social media search" in result.output
    assert "run" in result.output
    assert "search" in result.output
    assert "verify" in result.output
    assert "tamper" in result.output
    assert "chain-status" in result.output


def test_cli_chain_status_local():
    """Validates chain-status command on local network."""
    runner = CliRunner()
    result = runner.invoke(cli, ["chain-status", "--network", "local"])
    assert result.exit_code == 0
    assert "LOCAL" in result.output
    assert "total_blocks" in result.output


def test_cli_run_verify_tamper_offline_flow():
    """Validates end-to-end execution of run, verify, and tamper in offline demo mode."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = os.path.join(tmpdir, "out")
        chain_file = os.path.join(tmpdir, "test_local_chain.json")
        image_path = "tests/fixtures/person1_a.jpg"

        # 1. Run pipeline
        run_res = runner.invoke(
            cli,
            [
                "run",
                "--image", image_path,
                "--network", "local",
                "--offline-demo",
                "--out-dir", out_dir,
            ],
            env={"LOCAL_CHAIN_FILE": chain_file},
        )
        assert run_res.exit_code == 0, f"Run output: {run_res.output}"
        assert "STAGE 1: Face Identification" in run_res.output
        assert "STAGE 2: Genuine Social Media Search" in run_res.output
        assert "STAGE 3: Blockchain Anchoring" in run_res.output
        assert "Anchored on Local Simulated Blockchain" in run_res.output

        record_path = os.path.join(out_dir, "record.json")
        assert os.path.exists(record_path)

        # 2. Verify record
        verify_res = runner.invoke(
            cli,
            [
                "verify",
                "--record", record_path,
                "--network", "local",
            ],
            env={"LOCAL_CHAIN_FILE": chain_file},
        )
        assert verify_res.exit_code == 0, f"Verify output: {verify_res.output}"
        assert "Blockchain Verification Successful" in verify_res.output

        # 3. Tamper demo
        tamper_res = runner.invoke(
            cli,
            [
                "tamper",
                "--record", record_path,
                "--network", "local",
            ],
            env={"LOCAL_CHAIN_FILE": chain_file},
        )
        assert tamper_res.exit_code == 0, f"Tamper output: {tamper_res.output}"
        assert "TAMPERING DETECTED" in tamper_res.output
        assert "DEMO SUCCESS" in tamper_res.output
