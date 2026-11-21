#!/usr/bin/env bash
# Fail if anything tracked looks like an HF / Google / OpenAI-style key.
set -u
if git grep -nE "(hf_|AIza|sk-)[A-Za-z0-9]{10,}" -- . ; then
  echo "SECRET SCAN: possible secret found" >&2; exit 1
fi
echo "SECRET SCAN: clean"
