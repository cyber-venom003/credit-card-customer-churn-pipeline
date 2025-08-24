#!/usr/bin/env python3
"""
DVC Helper Script for Credit Card Churn Prediction Pipeline

This script provides convenient functions for common DVC operations
and helps maintain the data pipeline.
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"Error: {e}")
        if check:
            sys.exit(1)
        return e


def check_dvc_status():
    """Check the current DVC status."""
    print("🔍 Checking DVC status...")
    result = run_command(["dvc", "status"])
    if result.stdout.strip():
        print(result.stdout)
    else:
        print("✅ All data and pipelines are up to date!")


def regenerate_features():
    """Regenerate features using the DVC pipeline."""
    print("🔄 Regenerating features...")
    result = run_command(["dvc", "repro", "transform_features"])
    if result.returncode == 0:
        print("✅ Features regenerated successfully!")
        print("💡 Don't forget to commit the changes:")
        print("   git add dvc.lock dvc.yaml")
        print("   git commit -m 'Update features'")
    else:
        print("❌ Feature regeneration failed!")
        print(result.stderr)


def add_new_data_file(file_path: str):
    """Add a new data file to DVC tracking."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"📁 Adding {file_path} to DVC tracking...")
    result = run_command(["dvc", "add", file_path])
    if result.returncode == 0:
        print(f"✅ {file_path} added to DVC tracking!")
        print("💡 Don't forget to commit the .dvc file:")
        print(f"   git add {file_path}.dvc")
        print("   git commit -m 'Add new data file'")
    else:
        print(f"❌ Failed to add {file_path} to DVC!")


def list_tracked_files():
    """List all files tracked by DVC."""
    print("📋 Listing DVC tracked files...")
    result = run_command(["dvc", "list", "."])
    if result.stdout.strip():
        print(result.stdout)
    else:
        print("No files tracked by DVC.")


def setup_remote_storage(remote_name: str, remote_url: str):
    """Set up a new remote storage for DVC."""
    print(f"🌐 Setting up remote storage '{remote_name}' at {remote_url}...")
    
    # Add the remote
    result = run_command(["dvc", "remote", "add", remote_name, remote_url])
    if result.returncode != 0:
        print(f"❌ Failed to add remote: {result.stderr}")
        return
    
    # Set as default
    result = run_command(["dvc", "remote", "default", remote_name])
    if result.returncode == 0:
        print(f"✅ Remote '{remote_name}' set up successfully!")
        print("💡 You can now use 'dvc push' and 'dvc pull'")
    else:
        print(f"❌ Failed to set default remote: {result.stderr}")


def push_to_remote():
    """Push data to remote storage."""
    print("⬆️  Pushing data to remote storage...")
    result = run_command(["dvc", "push"])
    if result.returncode == 0:
        print("✅ Data pushed successfully!")
    else:
        print("❌ Failed to push data!")
        print(result.stderr)


def pull_from_remote():
    """Pull data from remote storage."""
    print("⬇️  Pulling data from remote storage...")
    result = run_command(["dvc", "pull"])
    if result.returncode == 0:
        print("✅ Data pulled successfully!")
    else:
        print("❌ Failed to pull data!")
        print(result.stderr)


def show_help():
    """Show help information."""
    print("""
🚀 DVC Helper Script for Credit Card Churn Prediction Pipeline

Usage: python dvc_helpers.py [command] [options]

Commands:
  status              Check DVC status
  regenerate          Regenerate features using DVC pipeline
  add <file>          Add a new data file to DVC tracking
  list                List all DVC tracked files
  remote <name> <url> Set up a new remote storage
  push                Push data to remote storage
  pull                Pull data from remote storage
  help                Show this help message

Examples:
  python dvc_helpers.py status
  python dvc_helpers.py regenerate
  python dvc_helpers.py add data/new_data.csv
  python dvc_helpers.py remote myremote s3://mybucket/path
  python dvc_helpers.py push
  python dvc_helpers.py pull
""")


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "status":
        check_dvc_status()
    elif command == "regenerate":
        regenerate_features()
    elif command == "add" and len(sys.argv) > 2:
        add_new_data_file(sys.argv[2])
    elif command == "list":
        list_tracked_files()
    elif command == "remote" and len(sys.argv) > 3:
        setup_remote_storage(sys.argv[2], sys.argv[3])
    elif command == "push":
        push_to_remote()
    elif command == "pull":
        pull_from_remote()
    elif command == "help":
        show_help()
    else:
        print(f"❌ Unknown command: {command}")
        show_help()


if __name__ == "__main__":
    main()
