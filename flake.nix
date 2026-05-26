{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable?shallow=1";
    flake-utils.url = "github:numtide/flake-utils?shallow=1";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;
        ffmpeg =
          if pkgs.stdenv.isLinux then
            pkgs.ffmpeg_8-headless.override {
              withHeadlessDeps = false;
              buildFfmpeg = true;
              buildFfprobe = true;
              buildAvcodec = true;
              buildAvdevice = true;
              buildAvfilter = true;
              buildAvformat = true;
              buildAvutil = true;
              buildSwresample = true;
              buildSwscale = true;
              withPixelutils = true;
              withHardcodedTables = true;
              withSafeBitstreamReader = true;
            }
          else
            pkgs.ffmpeg-headless;
      in
      {
        devShells.default =
          with pkgs;
          mkShell {
            packages = [
              nushell
              git
              git-lfs
              uv
              ytt
              just
              prek
              skim
              python
              ffmpeg
            ];

            shellHook =
              let
                libraryPath = lib.makeLibraryPath [
                  ffmpeg
                  stdenv.cc.cc.lib
                ];
                libraryPathEnvVar = if stdenv.isDarwin then "DYLD_FALLBACK_LIBRARY_PATH" else "LD_LIBRARY_PATH";
              in
              lib.strings.concatLines [
                "export UV_PYTHON=${python}/bin/python"
                "export UV_NO_MANAGED_PYTHON=1"
                "export UV_PYTHON_DOWNLOADS=never"
                "export ${libraryPathEnvVar}=${libraryPath}\${${libraryPathEnvVar}:+:${"$"}${libraryPathEnvVar}}"
              ];
          };
      }
    );
}
