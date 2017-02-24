all:*.pyc

%.pyc:%.py
	python -m compileall $<

