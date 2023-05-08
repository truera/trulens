if [ $# -eq 0 ]; then
    FILES="."
else
    FILES=${@:1}
fi
echo "REVIEWING FILES: ${FILES}"

# the configuration is at .isort.cfg
isort ${FILES}

yapf --style .style.yapf -r -i --verbose --parallel \
    ${FILES}
