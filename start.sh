#!/bin/sh

# Startup script for Dockerfile

# echo "Running migrations..."
# bun run db:migrate

# We'll use dev mode for now
echo "Starting application..."
bun run dev
