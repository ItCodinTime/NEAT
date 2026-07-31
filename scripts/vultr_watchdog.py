"""Poll a Vultr benchmark worker, sync artifacts, and delete it when done."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--ssh-key", required=True)
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--timeout-hours", type=float, default=18.0)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--hard-pending-charges", type=float, default=150.0)
    parser.add_argument("--max-polls", type=int, default=None)
    return parser.parse_args()


def ssh(args: argparse.Namespace, command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-i",
            args.ssh_key,
            f"root@{args.host}",
            command,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def rsync(args: argparse.Namespace) -> int:
    local = Path(args.local_dir)
    local.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "rsync",
            "-az",
            "--delete",
            "-e",
            (
                "ssh -o StrictHostKeyChecking=no "
                "-o UserKnownHostsFile=/dev/null "
                f"-i {args.ssh_key}"
            ),
            f"root@{args.host}:{args.remote_dir}/",
            str(local) + "/",
        ],
        check=False,
    )
    return result.returncode


def vultr_request(path: str, *, method: str = "GET") -> dict:
    token = os.environ["VULTR_API_KEY"]
    req = urllib.request.Request(
        "https://api.vultr.com/v2" + path,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        if response.status == 204:
            return {}
        return json.load(response)


def delete_instance(instance_id: str) -> None:
    vultr_request(f"/instances/{instance_id}", method="DELETE")


def pending_charges() -> float:
    payload = vultr_request("/account")
    return float(payload.get("account", {}).get("pending_charges") or 0.0)


def main() -> None:
    args = parse_args()
    started = time.time()
    deadline = started + args.timeout_hours * 3600
    local = Path(args.local_dir)
    local.mkdir(parents=True, exist_ok=True)
    state_path = local / "watchdog-state.jsonl"

    def record(event: str, **fields: object) -> None:
        row = {
            "event": event,
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        row.update(fields)
        with state_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    record("started", instance_id=args.instance_id, host=args.host)
    deleted = False
    try:
        polls = 0
        while True:
            polls += 1
            charges = pending_charges()
            status = ssh(
                args,
                (
                    f"test -f {args.remote_dir}/DONE && echo DONE || true; "
                    f"tail -n 40 {args.remote_dir}/status.txt 2>/dev/null || true; "
                    f"find {args.remote_dir} -name result.json 2>/dev/null | wc -l"
                ),
            )
            record(
                "poll",
                pending_charges=charges,
                ssh_returncode=status.returncode,
                remote_status=status.stdout[-4000:],
            )
            done = "DONE" in status.stdout.splitlines()
            timed_out = time.time() >= deadline
            over_budget = charges >= args.hard_pending_charges
            if done or timed_out or over_budget:
                sync_code = rsync(args)
                record(
                    "sync",
                    rsync_returncode=sync_code,
                    done=done,
                    timed_out=timed_out,
                    over_budget=over_budget,
                )
                delete_instance(args.instance_id)
                deleted = True
                record("deleted", instance_id=args.instance_id)
                break
            if args.max_polls is not None and polls >= args.max_polls:
                record("max_polls_reached", polls=polls)
                break
            time.sleep(args.poll_seconds)
    finally:
        if not deleted:
            record("exiting_without_delete")


if __name__ == "__main__":
    main()
