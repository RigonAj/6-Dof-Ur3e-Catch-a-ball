from setuptools import find_packages, setup

package_name = "ur3e_rollout_replay"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="rigon",
    maintainer_email="rigon@example.com",
    description="Validate and replay Isaac rollout joint targets on a UR3e.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ur3e_replay_validate = ur3e_rollout_replay.validate:main",
            "ur3e_replay_send = ur3e_rollout_replay.send:main",
        ],
    },
)
