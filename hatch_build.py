import logging
from collections.abc import Generator
from importlib import resources
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, final, override

from grpc_tools import protoc
from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from protoletariat.fdsetgen import Raw

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@final
class BuildYaakIdlProtosHook(BuildHookInterface):  # pyright: ignore[reportMissingTypeArgument]
    PLUGIN_NAME = "build-yaak-idl-protos"  # pyright: ignore[reportIncompatibleUnannotatedOverride]

    YAAK_IDL_PROTO_PATH = (
        Path(__file__).resolve().parent
        / "src"
        / "rbyte"
        / "io"
        / "yaak"
        / "idl-repo"
        / "intercom"
        / "proto"
    )
    YAAK_IDL_PYTHON_OUT = (
        Path(__file__).resolve().parent / "src" / "rbyte" / "io" / "yaak" / "proto"
    )
    YAAK_IDL_PROTOS = ("can.proto", "sensor.proto")

    @override
    def clean(self, versions: list[str]) -> None:
        for path in self._get_yaak_idl_proto_paths():
            if path.exists():
                logger.warning("removing %s", path)
                path.unlink()

    @override
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        self._build_yaak_idl_protos()

    @classmethod
    def _get_yaak_idl_proto_paths(cls) -> Generator[Path, Any, None]:
        for proto in cls.YAAK_IDL_PROTOS:
            name, *_ = proto.split(".", maxsplit=1)
            for ext in (".py", ".pyi"):
                yield (cls.YAAK_IDL_PYTHON_OUT / f"{name}_pb2").with_suffix(ext)

    @classmethod
    def _build_yaak_idl_protos(cls) -> None:
        with NamedTemporaryFile() as descriptor_set_out:
            protoc_cmd = [
                "grpc_tools.protoc",
                f"--proto_path={resources.files('grpc_tools') / '_proto'}",
                f"--proto_path={cls.YAAK_IDL_PROTO_PATH}",
                f"--python_out={cls.YAAK_IDL_PYTHON_OUT}",
                f"--pyi_out={cls.YAAK_IDL_PYTHON_OUT}",
                f"--descriptor_set_out={descriptor_set_out.name}",
                *cls.YAAK_IDL_PROTOS,
            ]

            if protoc.main(protoc_cmd) != 0:  # pyright: ignore[reportUnknownMemberType]
                msg = f"error: {protoc_cmd} failed"
                raise RuntimeError(msg)

            Raw(descriptor_set_out.read()).fix_imports(
                python_out=cls.YAAK_IDL_PYTHON_OUT,
                create_package=False,
                overwrite_callback=cls._overwrite_callback,
                module_suffixes=["_pb2.py", "_pb2.pyi"],
                exclude_imports_glob=["google/protobuf/*"],
            )

    @staticmethod
    def _overwrite_callback(file: Path, text: str) -> None:
        logger.warning("overwriting %s", file)
        _ = file.write_text(text)
