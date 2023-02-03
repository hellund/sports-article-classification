set -eu

doccano webserver --port 8000 &
doccano task &
start http://127.0.0.1:8000 &
cd src/annotation
flask --app active_learning run
