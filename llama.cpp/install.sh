#!/bin/bash

LINK_PREFIX="$(cd "$(dirname "$0")" && pwd)"
LINK_FILES=$(cd "$LINK_PREFIX" && ls build-* llama-*.sh 2>/dev/null | tr '\n' ' ')

echo "Creating symbolic links..."
for link in $LINK_FILES; do
    ln -sf "$LINK_PREFIX/$link" "$HOME/bin/$link"
done

echo "Verifying symbolic links..."
for link in $LINK_FILES; do
    if [ -L "$HOME/bin/$link" ] && [ -e "$HOME/bin/$link" ]; then
        target=$(readlink -f "$HOME/bin/$link")
        expected="$LINK_PREFIX/$link"
        if [ "$target" = "$expected" ]; then
            echo "Symbolic link $link created successfully, pointing to $target"
        else
            echo "Error: Symbolic link $link points to $target, expected $expected"
            exit 1
        fi
    else
        echo "Error: Symbolic link $link was not created or is broken"
        exit 1
    fi
done
