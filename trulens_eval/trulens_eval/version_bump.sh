# Function to bump the version
bump_version() {
    local major=$1
    local minor=$2
    local patch=$3

    # Determine OS and prepare SED in-place extension accordingly
    local SED_EXT="-i"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS requires an empty string argument after -i
        SED_EXT="-i ''"
    else
        SED_EXT="-i"
    fi

    # Replace the version tuple in __init__.py
    eval "sed $SED_EXT 's/__version_info__ = .*/__version_info__ = ($major, $minor, $patch)/' trulens_eval/__init__.py"
}

# Bump the version with the provided arguments
bump_version $1 $2 $3