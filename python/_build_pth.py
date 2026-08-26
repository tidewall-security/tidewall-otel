"""Build backend that adds ``tidewall_otel.pth`` to the wheel.

`backend-path` makes this module importable during an isolated build; it does
NOT make its functions PEP 517 hooks, because the frontend calls them on the
NAMED backend. So this module IS the backend, delegating everything to
setuptools and adding one file on the two build paths.

THE ARCHIVE CANNOT SIMPLY BE APPENDED TO. setuptools generates
``.dist-info/RECORD`` before this runs, so an appended file is unrecorded: a
permissive installer will install it, but the wheel is no longer internally
consistent, `pip uninstall` does not know to remove it, and integrity tooling
reports a malformed artifact. Appending a corrected RECORD is no better -- the
archive would then contain two members of that name. The wheel is therefore
REBUILT: every original member copied except RECORD, the ``.pth`` added, and a
fresh RECORD computed over the result.
"""

from setuptools import build_meta as _orig

# Every hook delegates. Only the two build hooks add the .pth.
prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
get_requires_for_build_sdist = _orig.get_requires_for_build_sdist
get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
get_requires_for_build_editable = _orig.get_requires_for_build_editable
prepare_metadata_for_build_editable = _orig.prepare_metadata_for_build_editable
build_sdist = _orig.build_sdist

_PTH_NAME = "tidewall_otel.pth"
#: A .pth line beginning `import` is the only form Python executes.
_PTH_BODY = b"import tidewall_otel_bootstrap\n"


def _record_line(arcname: str, data: bytes) -> str:
    """RECORD format: path,sha256=<urlsafe-b64>,size -- padding STRIPPED."""
    import base64
    import hashlib

    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{arcname},sha256={digest},{len(data)}\n"


def _inject_pth(wheel_directory: str, name: str) -> str:
    import os
    import zipfile
    from pathlib import Path

    path = Path(wheel_directory) / name
    with zipfile.ZipFile(path) as archive:
        members = [(info, archive.read(info.filename)) for info in archive.infolist()]

    record_name = next(info.filename for info, _ in members
                       if info.filename.endswith(".dist-info/RECORD"))

    rebuilt = path.parent / (path.name + ".tmp")
    with zipfile.ZipFile(rebuilt, "w", zipfile.ZIP_DEFLATED) as out:
        lines = []
        for info, data in members:
            if info.filename == record_name:
                continue                        # regenerated below
            out.writestr(info, data)            # preserves timestamps and mode
            lines.append(_record_line(info.filename, data))

        out.writestr(_PTH_NAME, _PTH_BODY)
        lines.append(_record_line(_PTH_NAME, _PTH_BODY))

        # RECORD lists itself with NO hash and NO size: it cannot contain its
        # own digest.
        lines.append(f"{record_name},,\n")
        out.writestr(record_name, "".join(lines))

    os.replace(rebuilt, path)
    return name


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    name = _orig.build_wheel(wheel_directory, config_settings, metadata_directory)
    return _inject_pth(wheel_directory, name)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    name = _orig.build_editable(wheel_directory, config_settings, metadata_directory)
    return _inject_pth(wheel_directory, name)
