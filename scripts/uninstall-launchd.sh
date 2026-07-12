#!/bin/sh
set -e
AGENTS="$HOME/Library/LaunchAgents"
for name in com.forecast-portfolio.mark com.forecast-portfolio.resolve; do
  launchctl unload "$AGENTS/$name.plist" 2>/dev/null || true
  rm -f "$AGENTS/$name.plist"
  echo "removed $name"
done
