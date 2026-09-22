# Usage: make <target> [EXP=<experiment module name>] [FILE=<submission csv>]
# Windows without make: run the python commands shown on each line.
PY ?= python

.PHONY: setup test smoke diagnose folds lock unlock quick full board blend sub validate

setup:     ; $(PY) -m pip install -r requirements.txt
test:      ; $(PY) -m pytest -q tests
smoke:     ; bash scripts/smoke_test.sh
diagnose:  ; $(PY) -m src.folds diagnose
folds:     ; $(PY) -m src.folds build
lock:      ; chmod -R a-w data/raw data/folds.csv config.yaml
unlock:    ; chmod -R u+w data/raw data/folds.csv config.yaml
quick:     ; QUICK=1 $(PY) -m experiments.$(EXP)
full:      ; $(PY) -m experiments.$(EXP)
board:     ; $(PY) -m src.board
blend:     ; $(PY) -m src.blend --top 8
sub:       ; $(PY) -m src.make_submission --exp $(EXP)
validate:  ; $(PY) -m src.validate $(FILE)
