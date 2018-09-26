SRC_ROOT = sklearn_seco/

default: check

check:
	flake8 "${SRC_ROOT}"
	mypy --ignore-missing-imports "${SRC_ROOT}"

test:
	pytest --durations 0 --cov "${SRC_ROOT}"

package: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf build/ dist/ "${SRC_ROOT}/sklearn_seco.egg-info/"

