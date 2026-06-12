from __future__ import annotations

import threading

UR_XACRO_PATH = "/opt/ros/humble/share/ur_description/urdf/ur.urdf.xacro"
UR_DESCRIPTION_SHARE = "/opt/ros/humble/share/ur_description"


class UrdfCache:
    """Holds the robot URDF.

    Preferred source is the /robot_description topic published by the running
    driver (it includes the extracted kinematics calibration). When the driver
    is not running, falls back to generating a default-kinematics URDF with
    xacro so the 3D viewer still works.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._topic_urdf: str | None = None
        self._xacro_urdf: str | None = None

    def set_from_topic(self, urdf_xml: str) -> None:
        with self._lock:
            self._topic_urdf = urdf_xml

    def get(self) -> tuple[str, str]:
        """Returns (urdf_xml, source) where source is "topic" or "xacro"."""
        with self._lock:
            if self._topic_urdf is not None:
                return self._topic_urdf, "topic"
            if self._xacro_urdf is not None:
                return self._xacro_urdf, "xacro"

        generated = _generate_with_xacro()
        with self._lock:
            self._xacro_urdf = generated
            if self._topic_urdf is not None:
                return self._topic_urdf, "topic"
            return self._xacro_urdf, "xacro"


def _generate_with_xacro() -> str:
    import xacro

    document = xacro.process_file(UR_XACRO_PATH, mappings={"ur_type": "ur3e", "name": "ur"})
    return document.toxml()
