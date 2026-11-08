#!/usr/bin/env bash
# Installs a post-commit hook that prints the git monitor table after every commit.
set -e
ROOT="$(git rev-parse --show-toplevel)"
HOOK="$ROOT/.git/hooks/post-commit"
cat > "$HOOK" <<'EOF'
#!/usr/bin/env bash
python3 "$(git rev-parse --show-toplevel)/scripts/git_monitor.py" || true
EOF
chmod +x "$HOOK"
echo "post-commit hook installed: $HOOK"
