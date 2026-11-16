#!/bin/zsh
# stop the test server listening on $PORT (default 8767): the python process that owns the port, then wait until free
PORT=${PORT:-8767}; P=$(lsof -tnP -iTCP:$PORT -sTCP:LISTEN)
[ -z "$P" ] && { echo "nothing listening on $PORT"; exit 0; }
echo "stopping test server pid $P ($(ps -o command= -p $P | cut -c1-90))"; kill $P
for i in $(seq 1 30); do lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 || { echo stopped; exit 0; }; sleep 1; done
echo "still listening"; exit 1
