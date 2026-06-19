{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable?shallow=1";
  };

  outputs = { nixpkgs, ... }: {
    devShells.x86_64-linux.default =
      with import nixpkgs { system = "x86_64-linux"; };
      let
        uvLibraryPath = lib.makeLibraryPath [
          stdenv.cc.cc.lib
          zlib
          openssl
          libffi
          ffmpeg
        ];
        uvWrapped = writeShellScriptBin "uv" ''
          export LD_LIBRARY_PATH="${uvLibraryPath}''${NIX_LD_LIBRARY_PATH:+:''${NIX_LD_LIBRARY_PATH}}"
          exec ${lib.getExe uv} "$@"
        '';
      in
      mkShell {
        packages = [
          nushell
          git
          git-lfs
          uvWrapped
          ytt
          just
          prek
          skim
          ffmpeg
        ];

        env = {
          UV_PYTHON = "${python312}/bin/python";
          UV_NO_MANAGED_PYTHON = "1";
          UV_PYTHON_DOWNLOADS = "never";
        };
      };

    devShells.aarch64-darwin.default =
      with import nixpkgs { system = "aarch64-darwin"; };
      let
        uvLibraryPath = lib.makeLibraryPath [
          stdenv.cc.cc.lib
          zlib
          openssl
          libffi
          ffmpeg
        ];
        uvWrapped = writeShellScriptBin "uv" ''
          export DYLD_FALLBACK_LIBRARY_PATH="${uvLibraryPath}''${DYLD_FALLBACK_LIBRARY_PATH:+:''${DYLD_FALLBACK_LIBRARY_PATH}}"
          exec ${lib.getExe uv} "$@"
        '';
      in
      mkShell {
        packages = [
          nushell
          git
          git-lfs
          uvWrapped
          ytt
          just
          prek
          skim
          ffmpeg
        ];

        env = {
          UV_PYTHON = "${python312}/bin/python";
          UV_NO_MANAGED_PYTHON = "1";
          UV_PYTHON_DOWNLOADS = "never";
        };
      };
  };
}
