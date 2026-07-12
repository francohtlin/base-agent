#!/bin/sh
# Install (or reinstall) the marking/resolution launchd agents.
# Marking every 6h is what feeds the IC metric; resolution realizes P&L daily.
# Neither job needs an Anthropic API key - they read public market data only.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AGENTS="$HOME/Library/LaunchAgents"
mkdir -p "$AGENTS"
for name in com.forecast-portfolio.mark com.forecast-portfolio.resolve; do
  launchctl unload "$AGENTS/$name.plist" 2>/dev/null || true
  cp "$DIR/$name.plist" "$AGENTS/"
  launchctl load "$AGENTS/$name.plist"
  echo "loaded $name"
done
echo "Uninstall with: scripts/uninstall-launchd.sh"
