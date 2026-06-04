#!/bin/bash

# Check if the version is provided as an argument
if [ -z "$1" ]; then
  echo "Error: No tag version provided."
  echo "Usage: sh update_tag.sh <tag-version>"
  exit 1
fi

TAG_VERSION=$1

echo "Updating tag: $TAG_VERSION"

# Delete the local tag if it exists
if git rev-parse "$TAG_VERSION" >/dev/null 2>&1; then
  git tag -d $TAG_VERSION
  echo "Deleted local tag: $TAG_VERSION"
else
  echo "Local tag $TAG_VERSION does not exist, continuing..."
fi

# Delete the remote tag if it exists
if git ls-remote --tags origin | grep -q "refs/tags/$TAG_VERSION"; then
  git push origin --delete $TAG_VERSION
  echo "Deleted remote tag: $TAG_VERSION"
else
  echo "Remote tag $TAG_VERSION does not exist, continuing..."
fi

# Create a new local tag
git tag -a $TAG_VERSION -m "Updated tag $TAG_VERSION"
if [ $? -ne 0 ]; then
  echo "Failed to create local tag $TAG_VERSION"
  exit 1
fi

# Push the new tag to the remote repository
git push origin $TAG_VERSION
if [ $? -ne 0 ]; then
  echo "Failed to push tag $TAG_VERSION to remote"
  exit 1
fi

echo "Tag $TAG_VERSION successfully updated!"
