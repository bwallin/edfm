from fabric.api import task, hosts, local, run, cd, lcd, settings

@task
def flake8():
    local('flake8 --config .flake8rc *.py **/*.py --verbose')


@task
def unit_test():
    local('nosetests --verbose')


@task
def test():
    flake8()
    unit_test()
