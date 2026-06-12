from setuptools import find_packages, setup

package_name = "ur3e_web_ui"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    # setuptools 59.6 (Ubuntu 22.04) does not expand "**" globs: list subdirs explicitly.
    package_data={
        package_name: [
            "static/*.html",
            "static/css/*.css",
            "static/js/*.js",
        ],
    },
    include_package_data=True,
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="rigon",
    maintainer_email="rigon@example.com",
    description="Browser UI for the UR3e: live 3D model, jog control, TCP pose, and rollout replay.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ur3e_web_ui = ur3e_web_ui.app:main",
        ],
    },
)
